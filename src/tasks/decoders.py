"""Decoders that interface between targets and model."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import src.models.nn.utils as U
import src.utils as utils
from src.models.sequence.backbones.model import SequenceModel


class Decoder(nn.Module):
    """Abstract class defining the interface for Decoders.

    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

        print(d_output, "d_output in SequenceDecoder")

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)

class NDDecoder(Decoder):
    """Decoder for single target (e.g. classification or regression)."""
    def __init__(
        self, d_model, d_output=None, mode="pool"
    ):
        super().__init__()

        assert mode in ["pool", "full"]
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        self.mode = mode

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.mode == 'pool':
            x = reduce(x, 'b ... h -> b h', 'mean')
        x = self.output_transform(x)
        return x

class StateDecoder(Decoder):
    """Use the output state to decode (useful for stateful models such as RNNs or perhaps Transformer-XL if it gets implemented."""

    def __init__(self, d_model, state_to_tensor, d_output):
        super().__init__()
        self.output_transform = nn.Linear(d_model, d_output)
        self.state_transform = state_to_tensor

    def forward(self, x, state=None):
        return self.output_transform(self.state_transform(state))


class RetrievalHead(nn.Module):
    def __init__(self, d_input, d_model, n_classes, nli=True, activation="relu"):
        super().__init__()
        self.nli = nli

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise NotImplementedError

        if (
            self.nli
        ):  # Architecture from https://github.com/mlpen/Nystromformer/blob/6539b895fa5f798ea0509d19f336d4be787b5708/reorganized_code/LRA/model_wrapper.py#L74
            self.classifier = nn.Sequential(
                nn.Linear(4 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, n_classes),
            )
        else:  # Head from https://github.com/google-research/long-range-arena/blob/ad0ff01a5b3492ade621553a1caae383b347e0c1/lra_benchmarks/models/layers/common_layers.py#L232
            self.classifier = nn.Sequential(
                nn.Linear(2 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, d_model // 2),
                activation_fn,
                nn.Linear(d_model // 2, n_classes),
            )

    def forward(self, x):
        """
        x: (2*batch, dim)
        """
        outs = rearrange(x, "(z b) d -> z b d", z=2)
        outs0, outs1 = outs[0], outs[1]  # (n_batch, d_input)
        if self.nli:
            features = torch.cat(
                [outs0, outs1, outs0 - outs1, outs0 * outs1], dim=-1
            )  # (batch, dim)
        else:
            features = torch.cat([outs0, outs1], dim=-1)  # (batch, dim)
        logits = self.classifier(features)
        return logits


class RetrievalDecoder(Decoder):
    """Combines the standard FeatureDecoder to extract a feature before passing through the RetrievalHead."""

    def __init__(
        self,
        d_input,
        n_classes,
        d_model=None,
        nli=True,
        activation="relu",
        *args,
        **kwargs
    ):
        super().__init__()
        if d_model is None:
            d_model = d_input
        self.feature = SequenceDecoder(
            d_input, d_output=None, l_output=0, *args, **kwargs
        )
        self.retrieval = RetrievalHead(
            d_input, d_model, n_classes, nli=nli, activation=activation
        )

    def forward(self, x, state=None, **kwargs):
        x = self.feature(x, state=state, **kwargs)
        x = self.retrieval(x)
        return x

class PackedDecoder(Decoder):
    def forward(self, x, state=None):
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


class S4Decoder(Decoder):
    """S4-based decoder head for sequence classification.

    This module applies an additional S4 `SequenceModel` stack on top of the
    backbone outputs and projects the pooled representation to the target
    dimensionality (e.g., intent classes for FSC).

    The constructor receives the backbone output dimension followed by the
    dataset output dimension via the registry plumbing. Additional keyword
    arguments configure the inner S4 stack in the same manner as existing
    sequence backbones.
    """

    def __init__(
        self,
        encoder_dim: int,
        num_classes: int,
        *,
    d_model: Optional[int] = None,
        aggregate: str = "mean",
        lengths_key: str = "lengths",
        input_dropout: float = 0.0,
    dropinp: Optional[float] = None,
        layer=None,
        residual=None,
        norm=None,
        pool=None,
        prenorm: bool = True,
        n_layers: int = 1,
        n_repeat: int = 1,
        dropout: float = 0.0,
        tie_dropout: bool = False,
        transposed: bool = False,
        track_norms: bool = False,
        **sequence_kwargs,
    ):
        super().__init__()

        if layer is None:
            raise ValueError("S4Decoder requires a 'layer' configuration for the inner SequenceModel.")

        self.aggregate = aggregate.lower()
        self.lengths_key = lengths_key
        self.num_classes = num_classes

        effective_d_model = d_model or encoder_dim
        dropinp = input_dropout if dropinp is None else dropinp

        if effective_d_model != encoder_dim:
            self.input_proj = nn.Linear(encoder_dim, effective_d_model)
        else:
            self.input_proj = nn.Identity()

        self.sequence_model = SequenceModel(
            d_model=effective_d_model,
            n_layers=n_layers,
            transposed=transposed,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            n_repeat=n_repeat,
            layer=layer,
            residual=residual,
            norm=norm,
            pool=pool,
            track_norms=track_norms,
            dropinp=dropinp,
            **sequence_kwargs,
        )

        self.output = nn.Linear(self.sequence_model.d_output, num_classes)

    def forward(self, x, state=None, lengths=None, **kwargs):
        if lengths is None and self.lengths_key in kwargs:
            lengths = kwargs[self.lengths_key]

        x = self.input_proj(x)
        x, state = self.sequence_model(x, state=state)
        x = self._reduce_sequence(x, lengths)
        x = self.output(x)
        return x, {"state": state}

    def step(self, x, state=None, **kwargs):
        x = self.input_proj(x)
        x, state = self.sequence_model.step(x, state=state)
        x = self.output(x)
        return x, {"state": state}

    def _reduce_sequence(self, x, lengths):
        if self.aggregate == "none":
            return x

        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=x.device)
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
        else:
            mask = None

        if self.aggregate in {"mean", "avg"}:
            if mask is None:
                return x.mean(dim=1)
            masked = x * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return masked.sum(dim=1) / denom

        if self.aggregate == "sum":
            return x.sum(dim=1) if mask is None else (x * mask.unsqueeze(-1)).sum(dim=1)

        if self.aggregate == "max":
            if mask is None:
                return x.max(dim=1).values
            masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return masked.max(dim=1).values

        if self.aggregate in {"last", "final"}:
            if mask is None or lengths is None:
                return x[:, -1]
            idx = (lengths - 1).clamp(min=0)
            return x[torch.arange(x.size(0), device=x.device), idx]

        if self.aggregate == "first":
            return x[:, 0]

        if self.aggregate in {"cls", "start"}:
            return x[:, 0]

        raise ValueError(f"Unknown aggregation mode '{self.aggregate}' for S4Decoder.")


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
    "nd": NDDecoder,
    "retrieval": RetrievalDecoder,
    "state": StateDecoder,
    "pack": PackedDecoder,

    "s4": S4Decoder,
}
model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "s4": ["d_output"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    # TODO rename d_output to n_classes?
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "s4": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )

    print("***Dataset args:***", dataset_args)

    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
