import torch
import torchaudio
import sys
import torch.nn as nn
import pandas as pd
from src.models.sequence.backbones.model import SequenceModel

NUM_CLASSES = 6
CKPT_PATH = "testing/accuracy.ckpt"
CSV_PATH = "data/fluent_speech_commands_dataset/data/train_data.csv"  # Change if needed

def get_action_labels(csv_path):
    df = pd.read_csv(csv_path)
    actions = sorted(df["action"].unique())
    ##EMAD: Print the actions for debugging
    print(actions)
    return actions

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

encoder = nn.Linear(1, 128)
decoder = nn.Linear(128, NUM_CLASSES)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)

# Load encoder weights
if "encoder.0.weight" in ckpt["state_dict"] and "encoder.0.bias" in ckpt["state_dict"]:
    encoder.weight.data.copy_(ckpt["state_dict"]["encoder.0.weight"])
    encoder.bias.data.copy_(ckpt["state_dict"]["encoder.0.bias"])
    print("Loaded encoder weights.")
else:
    print("WARNING: Encoder weights not found in checkpoint. Using random encoder weights!")

# Load decoder weights
if "decoder.0.output_transform.weight" in ckpt["state_dict"] and "decoder.0.output_transform.bias" in ckpt["state_dict"]:
    decoder.weight.data.copy_(ckpt["state_dict"]["decoder.0.output_transform.weight"])
    decoder.bias.data.copy_(ckpt["state_dict"]["decoder.0.output_transform.bias"])
    print("Loaded decoder weights.")
else:
    print("WARNING: Decoder weights not found in checkpoint. Using random decoder weights!")

model.eval()
encoder.eval()
decoder.eval()

def preprocess_wav(wav_path, length=16000):
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0)  # mono
    if len(waveform) < length:
        waveform = torch.nn.functional.pad(waveform, (0, length - len(waveform)))
    else:
        waveform = waveform[:length]
    waveform = waveform.unsqueeze(0).unsqueeze(-1)  # (1, length, 1)
    return waveform

def predict_action(wav_path, actions):
    x = preprocess_wav(wav_path)
    with torch.no_grad():
        x_encoded = encoder(x)     # (1, length, 128)
        logits = model(x_encoded)[0]
        if logits.ndim == 3:
            logits = logits[:, -1]
        logits = decoder(logits)   # (1, NUM_CLASSES)
        pred = logits.argmax(dim=-1).item()
    if pred < len(actions):
        return actions[pred]
    else:
        return f"Unknown action index: {pred}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <wav_file>")
        sys.exit(1)
    wav_file = sys.argv[1]
    actions = get_action_labels(CSV_PATH)
    action = predict_action(wav_file, actions)
    print(f"Predicted action: {action}")