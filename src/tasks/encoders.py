"""Encoders that interface between input data and model."""

import datetime
import math
from typing import ForwardRef

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config
from src.models.sequence.backbones.block import SequenceResidualBlock
from src.models.nn import Normalization

class Encoder(nn.Module):
    """Encoder abstraction.

    Accepts a tensor and optional kwargs. Outside of the main tensor, all other arguments should be kwargs.
    Returns a tensor and optional kwargs.
    Encoders are combined via U.PassthroughSequential which passes these kwargs through in a pipeline. The resulting kwargs are accumulated and passed into the model backbone.

    """

    def forward(self, x, **kwargs):
        """
        x: input tensor
        *args: additional info from the dataset (e.g. sequence lengths)

        Returns:
        y: output tensor
        *args: other arguments to pass into the model backbone
        """
        return x, {}



# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoder(Encoder):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoder(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=16384, pe_init=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if pe_init is not None:
            self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
            nn.init.normal_(self.pe, 0, pe_init)
            # self.pe = pe.unsqueeze(1)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0.0, max_len).unsqueeze(1)
            div_term = torch.exp(
                -math.log(10000.0) * torch.arange(0.0, d_model, 2.0) / d_model
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        self.attn_mask = None

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            lens: actual lengths of sequences
        Shape:
            x: [l_sequence, n_batch, d_model]
            Returns: [l_sequence, n_batch, d_model]
            attn_mask: [l_sequence, l_sequence]
            padding_mask:
        """

        x = x + self.pe[: x.size(-2)]
        return self.dropout(x)


class ClassEmbedding(Encoder):
    # Should also be able to define this by subclassing Embedding
    def __init__(self, n_classes, d_model):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, d_model)

    def forward(self, x, y):
        x = x + self.embedding(y).unsqueeze(-2)  # (B, L, D)
        return x


class Conv1DEncoder(Encoder):
    def __init__(self, d_input, d_model, kernel_size=25, stride=1, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_input,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        # BLD -> BLD
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x


class ConvStackEncoder(Encoder):
    """Builds a configurable stack of 1D convolutions from raw waveform to model features.

    Constructor signature: ConvStackEncoder(d_input, d_model, layers)
    - d_input: dataset input channels (positional, provided by dataset)
    - d_model: final model feature dimension (positional, provided by model)
    - layers: list of dicts describing each conv layer. Each dict supports:
        out_channels (int), kernel_size (int), stride (int), padding (int or 'same'), activation (gelu/relu/none)

    The final layer's out_channels will be projected to d_model if it doesn't match.
    """

    def __init__(self, d_input, d_model, layers=None):
        super().__init__()
        if layers is None:
            raise AssertionError("layers must be provided and be a list-like of layer specs")

        # Convert ListConfig or other sequence types to a plain Python list
        try:
            layers_list = list(layers)
        except Exception:
            raise AssertionError("layers must be an iterable of layer spec dictionaries")

        modules = []
        in_ch = d_input
        # Normalize specs to plain dicts for safe access
        self.layer_specs = [dict(spec) if not isinstance(spec, dict) else spec for spec in layers_list]

        for i, spec in enumerate(self.layer_specs):
            out_ch = spec.get("out_channels")
            if out_ch is None:
                raise AssertionError(f"layer spec at index {i} missing 'out_channels'")
            k = spec.get("kernel_size", 3)
            s = spec.get("stride", 1)
            p = spec.get("padding", 0)

            block_modules = []
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
            block_modules.append(conv)

            norm_type = spec.get("norm", "batch")
            if norm_type == "batch":
                block_modules.append(nn.BatchNorm1d(out_ch))
            elif norm_type == "layer":
                # Use GroupNorm with 1 group as LayerNorm for Conv1d
                block_modules.append(nn.GroupNorm(num_groups=1, num_channels=out_ch))
            else:
                block_modules.append(nn.Identity())

            pool_kernel = spec.get("pool_kernel_size", 2)
            pool_stride = spec.get("pool_stride", 2)
            block_modules.append(nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride))

            act = spec.get("activation", None)
            if act is not None:
                if act == "gelu":
                    block_modules.append(nn.GELU())
                elif act == "relu":
                    block_modules.append(nn.ReLU())
                # add others if needed

            drop_p = spec.get("dropout", 0.0)
            if drop_p and float(drop_p) > 0.0:
                block_modules.append(nn.Dropout(p=float(drop_p)))

            modules.append(nn.Sequential(*block_modules))
            in_ch = out_ch

        # Final projection to d_model if necessary
        self.net = nn.ModuleList(modules)
        self.final_out = in_ch
        if self.final_out != d_model:
            # 1x1 conv to project channels to d_model
            self.proj = nn.Conv1d(self.final_out, d_model, kernel_size=1)
        else:
            self.proj = None

        # No need for extra norm layers; handled in block_modules above
        self.norm_layers = None

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)

        for li, block in enumerate(self.net):
            x_in = x
            x = block(x)
            # Per-block residual connection
            if x.shape == x_in.shape:
                x = x + x_in
            else:
                # Interpolate x_in to match x's sequence length if needed
                if x_in.shape[2] != x.shape[2]:
                    x_in_resized = F.interpolate(x_in, size=x.shape[2], mode="linear", align_corners=False)
                else:
                    x_in_resized = x_in
                proj_res = nn.Conv1d(x_in_resized.shape[1], x.shape[1], kernel_size=1).to(x.device)
                x = x + proj_res(x_in_resized)

        if self.proj is not None:
            x = self.proj(x)

        x = x.transpose(1, 2)  # (B, L, D)
        return x

class Conv2DEncoder(Encoder):
    """For encoding Mel-spectrograms or images into a sequence of patches/features.

    Arguments:
      - in_channels: input channels (usually 1 for Mel-spectrograms)
      - out_channels: output channels
      - kernel_size: tuple (height, width)
      - stride: tuple (height, width)
      - padding: tuple (height, width)
      - activation: activation function (gelu/relu/none)
      - norm: normalization type (batch/none)
      - pool: pooling type (max/none)
      - pool_kernel_size: tuple
      - pool_stride: tuple
      - dropout: float
    """
    def __init__(self, in_channels=1, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), activation="gelu", norm="batch", pool="max", pool_kernel_size=(2,2), pool_stride=(2,2), dropout=0.2):
        super().__init__()
        # Ensure compatibility with Hydra ListConfig and lists
        if not isinstance(kernel_size, tuple):
            kernel_size = tuple(list(kernel_size))
        if not isinstance(stride, tuple):
            stride = tuple(list(stride))
        if not isinstance(padding, tuple):
            padding = tuple(list(padding))
        if not isinstance(pool_kernel_size, tuple):
            pool_kernel_size = tuple(list(pool_kernel_size))
        if not isinstance(pool_stride, tuple):
            pool_stride = tuple(list(pool_stride))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels) if norm == "batch" else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride) if pool == "max" else nn.Identity()
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, 1, H, W) for Mel-spectrograms
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.act(x)
        x = self.dropout(x)
        # flatten spatial dims to sequence
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, seq_len, C)
        return x
    
class Conv2DStackEncoder(Encoder):
    """
    Stacked Conv2D layers for encoding 2D data like Mel-spectrograms.

    Args:
        in_channels: input channels
        d_model: output channels per layer and final output
        n_layers: number of conv layers
        kernel_size: int or tuple
        stride: int or tuple
        padding: int or tuple
        activation: activation function
        norm: normalization type
        pool: pooling type
        pool_kernel_size: int or tuple
        pool_stride: int or tuple
        dropout: float
    """
    def __init__(self, in_channels=1, d_model=256, n_layers=2, kernel_size=3, stride=1, padding=1, activation='gelu', norm='batch', pool='max', pool_kernel_size=2, pool_stride=2, dropout=0.0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(pool_kernel_size, int):
            pool_kernel_size = (pool_kernel_size, pool_kernel_size)
        if isinstance(pool_stride, int):
            pool_stride = (pool_stride, pool_stride)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else d_model,
                    d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            if norm == 'batch':
                self.layers.append(nn.BatchNorm2d(d_model))
            if activation == 'gelu':
                self.layers.append(nn.GELU())
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Identity())
            if dropout > 0:
                self.layers.append(nn.Dropout2d(dropout))

        if pool == 'max':
            # Ensure compatibility with Hydra ListConfig and lists
            if not isinstance(pool_kernel_size, tuple):
                pool_kernel_size = tuple(list(pool_kernel_size))
            if not isinstance(pool_stride, tuple):
                pool_stride = tuple(list(pool_stride))
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, seq_len, C)
        return x

class LayerEncoder(Encoder):
    """Use an arbitary SequenceModule layer"""

    def __init__(self, d_model, prenorm=False, norm='layer', layer=None):
        super().__init__()

        # Simple stack of blocks
        layer["transposed"] = False
        self.layer = SequenceResidualBlock(
            d_input=d_model,
            prenorm=prenorm,
            layer=layer,
            residual='R',
            norm=norm,
            pool=None,
        )

    def forward(self, x):
        x, _ = self.layer(x) # Discard state
        return x


class TimestampEmbeddingEncoder(Encoder):
    """
    General time encoder for Pandas Timestamp objects (encoded as torch tensors).
    See MonashDataset for an example of how to return time features as 'z's.
    """

    cardinalities = {
        'day': (1, 31),
        'hour': (0, 23),
        'minute': (0, 59),
        'second': (0, 59),
        'month': (1, 12),
        'year': (1950, 2010), # (1800, 3000) used to be (1970, datetime.datetime.now().year + 1) but was not enough for all datasets in monash
        'dayofweek': (0, 6),
        'dayofyear': (1, 366),
        'quarter': (1, 4),
        'week': (1, 53),
        'is_month_start': (0, 1),
        'is_month_end': (0, 1),
        'is_quarter_start': (0, 1),
        'is_quarter_end': (0, 1),
        'is_year_start': (0, 1),
        'is_year_end': (0, 1),
        'is_leap_year': (0, 1),
    }

    def __init__(self, d_model, table=False, features=None):
        super().__init__()
        self.table = table
        self.ranges = {k: max_val - min_val + 2 for k, (min_val, max_val) in self.cardinalities.items()} # padding for null included

        if features is None:
            pass
        else:
            self.cardinalities = {k: v for k, v in self.cardinalities.items() if k in features}

        if table:
            self.embedding = nn.ModuleDict({
                attr: nn.Embedding(maxval - minval + 2, d_model, padding_idx=0)
                for attr, (minval, maxval) in self.cardinalities.items()
            })
        else:
            self.embedding = nn.ModuleDict({
                attr: nn.Linear(1, d_model)
                for attr in self.cardinalities
            })



    def forward(self, x, timestamps=None):
        for attr in timestamps:
            mask = timestamps[attr] == -1
            timestamps[attr] = timestamps[attr] - self.cardinalities[attr][0]
            timestamps[attr][mask] = 0
            if self.table:
                x = x + self.embedding[attr](timestamps[attr].to(torch.long))
            else:
                x = x + self.embedding[attr]((2 * timestamps[attr] / self.ranges[attr] - 1).unsqueeze(-1))

            #x = x + self.embedding(timestamps[attr].to(torch.float)).unsqueeze(1)
        return x

# TODO is this used anymore?
class TSIndexEmbeddingEncoder(Encoder):
    """
    Embeds location of sample in the time series
    """

    def __init__(self, n_ts, d_model, table=True):
        super().__init__()

        self.table = table
        self.n_ts = n_ts
        if table:
            self.embedding = nn.Embedding(n_ts, d_model)
        else:
            # self.embedding = nn.Linear(1, d_model)
            # self.linear = nn.Linear(2 * d_model, d_model)

            self.linear = nn.Linear(d_model + 1, d_model)

    def forward(self, x, z=None, idxs=None):
        if self.table:
            x = x + self.embedding(idxs.to(torch.long)).unsqueeze(1)
        else:
            # x = self.linear(torch.cat([x, self.embedding((2 * idxs / self.n_ts - 1)[:, None, None]).repeat((1, x.shape[1], 1))], axis=-1))
            x = self.linear(torch.cat([x, ((2 * idxs / self.n_ts - 1)[:, None, None]).repeat((1, x.shape[1], 1))], axis=-1))
        #x = x + self.embedding(idxs.unsqueeze(1).to(torch.float)).unsqueeze(1)
        return x


class TimeEncoder(Encoder):
    def __init__(self, n_tokens_time, d_model, timeenc=0):
        super().__init__()

        self.timeenc = timeenc
        if self.timeenc == 0:
            self.encoders = nn.ModuleList(
                [nn.Embedding(v, d_model) for v in n_tokens_time]
            )
        else:
            self.encoders = nn.Linear(len(n_tokens_time), d_model)
        self.mask_embed = nn.Embedding(2, d_model)

    def forward(self, x, mark=None, mask=None):
        assert mark is not None and mask is not None, "Extra arguments should be returned by collate function"
        if self.timeenc == 0:
            assert mark.size(-1) == len(self.encoders)
            embeddings = [
                embed(z) for embed, z in zip(self.encoders, torch.unbind(mark, dim=-1))
            ]
            time_encode = torch.sum(torch.stack(embeddings), dim=0)
        else:
            time_encode = self.encoders(mark)
        mask_encode = self.mask_embed(mask.squeeze(-1))
        return x + time_encode + mask_encode  # (B, L, d_model)

class EEGAgeEncoder(Encoder):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)

    def forward(self, x, age=None):
        z = self.encoder(((age - 50.0) / 100.0).unsqueeze(1))
        return x + z.unsqueeze(1)

class PackedEncoder(Encoder):
    def forward(self, x, len_batch=None):
        assert len_batch is not None
        x = nn.utils.rnn.pack_padded_sequence(
            x, len_batch.cpu(), enforce_sorted=False, batch_first=True,
        )
        return x


class OneHotEncoder(Encoder):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        assert n_tokens <= d_model
        self.d_model = d_model

    def forward(self, x):
        return F.one_hot(x.squeeze(-1), self.d_model).float()


# TransformerEncoder for sequence feature extraction
class TransformerEncoder(Encoder):
    """Transformer encoder for sequence feature extraction (uses nn.TransformerEncoder)."""
    def __init__(self, d_input, d_model, n_layers=2, n_heads=4, dim_feedforward=256, dropout=0.1, activation="relu"):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers) #, dropout=dropout)

    def forward(self, x, **kwargs):
        # x: (B, L, D)
        x = self.input_proj(x)
        x = self.encoder(x)
        return x


class Conv3DPatchEncoder(Encoder):
    """For encoding 3D data (e.g. videos) into a sequence of patches.

    Arguments:
      - d_emb: dim of embedding output
      - filter_sizes: tuple, with ft, fh, fw
      - max_len: int, max seq len
    """
    def __init__(self, d_emb, filter_sizes, pos_enc=False, max_len=2352):
        self.pos_enc = pos_enc
        ft, fh, fw = filter_sizes

        super().__init__()
        assert len(filter_sizes) == 3

        self.encoder = nn.Conv3d(3, d_emb, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

    def forward(self, x):
        """
        x: shape = [b, c, t, h, w]

        Returns tuple with x, with new shape = [b, seq_len, c_out]
        """
        x = self.encoder(x)
        b, c, t, h, w = x.shape

        x = x.reshape([b, c, t*h*w])  # flatten spatial / temporal dim
        x = x.permute(0, 2, 1)  # permute the c and seq len for s4

        return x

class Conv2DPatchEncoder(Encoder):
    """For encoding images into a sequence of patches.

    Arguments:
      - d_input: dim of encoder input (data dimension)
      - d_model: dim of encoder output (model dimension)
      - filter_sizes: tuple with fh, fw
      - flat: if image is flattened from dataloader (like in cifar),
        then we need to reshape back to 2D before conv
    """

    def __init__(self, d_input, d_model, filter_sizes, flat=False):
        fh, fw = filter_sizes
        self.flat = flat

        super().__init__()
        assert len(filter_sizes) == 2

        self.encoder = nn.Conv2d(d_input, d_model, kernel_size=(fh, fw), stride=(fh, fw))

    def forward(self, x):
        """
        x shape = [b, h, w, c]
        Returns tuple with x, with new shape = [b, seq_len, c_out]
        """

        x = rearrange(x, 'b h w c -> b c h w')
        x = self.encoder(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class TextConditionalEncoder(Encoder):

    def __init__(self, vocab_size, d_model, n_layers, layer, reversal=False):
        super().__init__()
        # d_model = 2 * d_model
        self.reversal = reversal
        self.padding_idx = vocab_size - 1
        self.text_embedding = nn.Embedding(vocab_size, d_model)

        # Simple stack of blocks
        self.text_encoder = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                i_layer=i,
                prenorm=True,
                layer=layer,
                residual='R',
                norm='layer',
                pool=None,
                transposed=True,
            ) for i in range(n_layers)
        ])

        # self.output_linear = nn.Linear(d_model, d_model // 2)

        # self.norm = Normalization(d_model, transposed=True, _name_='layer')

    def forward(self, x, tokens=None, text_lengths=None):
        # Arguments must be in this order
        # lengths, tokens, text_lengths = args
        assert tokens is not None and text_lengths is not None


        # Calculate the text embedding
        text_embedding = self.text_embedding(tokens) # (B, L, D)
        text_embedding = text_embedding.transpose(1, 2) # (B, D, L)
        for layer in self.text_encoder:
            text_embedding, _ = layer(text_embedding)

            if self.reversal:
                # Reverse the sequence
                text_embedding = text_embedding.fliplr()
        # text_embedding = self.norm(text_embedding)
        text_embedding = text_embedding.transpose(1, 2)

        # Zero out the embedding for padding tokens
        mask = (tokens != self.padding_idx).unsqueeze(2)
        text_embedding = text_embedding * mask.float()

        # Calculate the mean embedding for each sequence (normalizing by appropriate token lengths)
        text_embedding = text_embedding.sum(dim=1) / text_lengths.float().unsqueeze(1)
        # text_embedding = self.output_linear(text_embedding)

        # Add the text embedding to the sequence embedding (for global conditioning)
        x = x + text_embedding.unsqueeze(1)

        return x



# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Encoder,
    "id": nn.Identity,
    "embedding": nn.Embedding,
    "linear": nn.Linear,
    "position": PositionalEncoder,
    "class": ClassEmbedding,
    "pack": PackedEncoder,
    "time": TimeEncoder,
    "onehot": OneHotEncoder,
    "conv1d": Conv1DEncoder,
    "eegage": EEGAgeEncoder,
    "patch3d": Conv3DPatchEncoder,
    "patch2d": Conv2DPatchEncoder,
    "textcond": TextConditionalEncoder,
    "timestamp_embedding": TimestampEmbeddingEncoder,
    "tsindex_embedding": TSIndexEmbeddingEncoder,
    "layer": LayerEncoder,
    
    "conv_stack": ConvStackEncoder,
    "conv2d": Conv2DEncoder,
    "conv2d_stack": Conv2DStackEncoder,
    "transformer": TransformerEncoder,
}
dataset_attrs = {
    "embedding": ["n_tokens"],
    "textcond": ["vocab_size"],
    "linear": ["d_input"],  # TODO make this d_data?
    "class": ["n_classes"],
    "time": ["n_tokens_time"],
    "onehot": ["n_tokens"],
    "conv1d": ["d_input"],
    "conv_stack": ["d_input"],
    "patch2d": ["d_input"],
    "tsindex_embedding": ["n_ts"],
}
model_attrs = {
    "embedding": ["d_model"],
    "textcond": ["d_model"],
    "linear": ["d_model"],
    "position": ["d_model"],
    "class": ["d_model"],
    "time": ["d_model"],
    "onehot": ["d_model"],
    "conv1d": ["d_model"],
    "conv_stack": ["d_model"],
    "patch2d": ["d_model"],
    "eegage": ["d_model"],
    "timestamp_embedding": ["d_model"],
    "tsindex_embedding": ["d_model"],
    "layer": ["d_model"],
}


def _instantiate(encoder, dataset=None, model=None):
    """Instantiate a single encoder"""
    if encoder is None:
        return None
    if isinstance(encoder, str):
        name = encoder
    else:
        name = encoder["_name_"]

    # Extract dataset/model arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))

    # Instantiate encoder
    obj = utils.instantiate(registry, encoder, *dataset_args, *model_args)
    return obj


def instantiate(encoder, dataset=None, model=None):
    encoder = utils.to_list(encoder)
    return U.PassthroughSequential(
        *[_instantiate(e, dataset=dataset, model=model) for e in encoder]
    )
