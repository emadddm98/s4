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

model = SequenceModel()

# encoder = nn.Linear(1, 128)
encoder = nn.Identity()
# decoder = nn.Linear(128, NUM_CLASSES)
decoder = SequenceModel()

ckpt = torch.load(CKPT_PATH, map_location="cuda")
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

def test_with_dummy_data():
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        device = "cpu"
    else:
        device = "cuda"
    dummy_input = torch.randn(1, 16000, 1).to(device)
    encoder.to(device)
    model.to(device)
    decoder.to(device)
    with torch.no_grad():
        x_encoded = encoder(dummy_input)
        logits = model(x_encoded)[0]
        if logits.ndim == 3:
            logits = logits[:, -1]
        logits = decoder(logits)
        print("Dummy logits shape:", logits.shape)
        if device == "cuda":
            print("CUDA memory summary:")
            print(torch.cuda.memory_summary())
        else:
            print("CPU test complete. No CUDA memory info.")

# Uncomment to run dummy data test
test_with_dummy_data()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <wav_file>")
        sys.exit(1)
    wav_file = sys.argv[1]
    actions = get_action_labels(CSV_PATH)
    action = predict_action(wav_file, actions)
    print(f"Predicted action: {action}")