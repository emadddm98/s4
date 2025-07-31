from src.models.sequence.backbones.model import SequenceModel

model = SequenceModel(
    d_model=128,
    n_layers=2,
    transposed=False,
    dropout=0.0,
    tie_dropout=True,
    prenorm=True,
    bidirectional=False,
    n_repeat=1,
    layer=[{
        'd_state': 32,
        'channels': 1,
        'bidirectional': False,
        'gate': None,
        'gate_act': 'id',
        'bottleneck': None,
        'activation': 'gelu',
        'mult_act': None,
        'final_act': 'glu',
        'postact': None,
        'initializer': None,
        'weight_norm': False,
        'tie_dropout': True,
        'layer': 'fftconv',
        'mode': 'nplr',
        'init': 'legs',
        'measure': None,
        'rank': 1,
        'dt_min': 0.001,
        'dt_max': 0.1,
        'dt_transform': 'softplus',
        'lr': {'dt': 0.001, 'A': 0.001, 'B': 0.001},
        'wd': 0.0,
        'n_ssm': 1,
        'drop_kernel': 0.0,
        'deterministic': False,
        'l_max': 16000,
        'verbose': True,
        'dropout': 0.0,
        'transposed': False,
        '_name_': 's4'
    }],
    residual='R',
    norm='layer',
    pool={'stride': 1, 'expand': None, '_name_': 'pool'},
    track_norms=True,
    dropinp=0.0
)

import torch

ckpt = torch.load("testing/accuracy.ckpt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()