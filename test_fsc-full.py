import os
# from src.dataloaders.datasets.fsc_full import get_fsc_datasets
from src.dataloaders.datasets.fsc_multilabel import FSCMultiLabelDataset
# from src.dataloaders.datasets.fsc import get_fsc_datasets
from src.dataloaders.datasets.fsc_randomsplit import get_fsc_datasets
import torch
import torchaudio

def test_torchaudio_fsc(root):
    dataset = torchaudio.datasets.FluentSpeechCommands(root, subset="train")
    print(f"torchaudio FSC train size: {len(dataset)}")
    sample = dataset[10]
    print("Sample tuple (all fields):")
    for i, value in enumerate(sample):
        print(f"  [{i}] {type(value)}: {value if not isinstance(value, torch.Tensor) else value.shape}")
    print("\nField mapping (by index):")
    print("  [0] waveform (Tensor)")
    print("  [1] sample_rate (int)")
    print("  [2] transcript or utterance_id (str)")
    print("  [3] speaker_id (str)")
    print("  [4] transcription (str)")
    print("  [5] action (str)")
    print("  [6] object (str)")
    print("  [7] location (str)")

def test_fsc_dataloader(root):
    train, val, test = get_fsc_datasets(root, mfcc=False)
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    print(f"Number of intents (classes): {len(train.intent2idx)}")
    print(f"Intent mapping: {train.intent2idx}")

    # Check a few samples
    for split, ds in zip(['train', 'val', 'test'], [train, val, test]):
        x, y = ds[0]
        # Find the intent string corresponding to the index
        intent_str = [k for k, v in ds.intent2idx.items() if v == y][0]
        # Get the relative path or utterance_id from the metadata
        rel_path = ds.meta.iloc[0]['path'] if hasattr(ds, 'meta') and 'path' in ds.meta.columns else 'N/A'
        print(f"{split} sample shape: {x.shape}, label (intent idx): {y}, intent string: {intent_str}, path: {rel_path}")
        assert isinstance(x, (list, tuple, type(x)))
        assert isinstance(y, int)
        if hasattr(ds, 'feature_max_length'):
            print(f"{split} feature_max_length: {ds.feature_max_length}")
        break  # Only check the first sample

    # Check label consistency
    assert train.intent2idx == val.intent2idx == test.intent2idx, "Intent mappings differ between splits!"

def test_fsc_multilabel_dataloader(root):
    """Test the multilabel FSC dataset (action, object, location)"""
    print("\n--- Testing FSC Multilabel Dataset ---")
    
    # Create train dataset first to get label mappings
    train = FSCMultiLabelDataset(root, split="train", mfcc=True, n_mfcc=64, n_mels=80)
    val = FSCMultiLabelDataset(root, split="valid", component_label2idx=train.label2idx)
    test = FSCMultiLabelDataset(root, split="test", component_label2idx=train.label2idx)
    
    print(f"Multilabel Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    
    # Print label mappings
    for label_type in ['action', 'object', 'location']:
        print(f"{label_type.capitalize()} labels ({len(train.label2idx[label_type])}): {list(train.label2idx[label_type].keys())}")
    
    # Test samples
    for split, ds in zip(['train', 'val', 'test'], [train, val, test]):
        x, labels = ds[0]
        action_idx, object_idx, location_idx = labels
        
        # Get string labels
        action_str = [k for k, v in ds.label2idx['action'].items() if v == action_idx][0]
        object_str = [k for k, v in ds.label2idx['object'].items() if v == object_idx][0]
        location_str = [k for k, v in ds.label2idx['location'].items() if v == location_idx][0]
        
        rel_path = ds.meta.iloc[0]['path'] if hasattr(ds, 'meta') and 'path' in ds.meta.columns else 'N/A'
        
        print(f"{split} sample shape: {x.shape}")
        print(f"  Labels (indices): action={action_idx}, object={object_idx}, location={location_idx}")
        print(f"  Labels (strings): action='{action_str}', object='{object_str}', location='{location_str}'")
        print(f"  Path: {rel_path}")
        
        # Assertions
        assert isinstance(x, torch.Tensor)
        assert isinstance(labels, tuple) and len(labels) == 3
        assert all(isinstance(label, int) for label in labels)
        break  # Only check first sample
    
    # Check label consistency across splits
    for label_type in ['action', 'object', 'location']:
        assert train.label2idx[label_type] == val.label2idx[label_type] == test.label2idx[label_type], f"{label_type} mappings differ between splits!"
    
    print("✅ Multilabel dataset test passed!")

if __name__ == "__main__":
    # Set this to your FSC dataset root directory
    FSC_ROOT = "/workspace/stm/s4/data/fluent_speech_commands_dataset"
    test_fsc_dataloader(FSC_ROOT)
    # test_fsc_multilabel_dataloader(FSC_ROOT)
    # print("\n--- Testing torchaudio.datasets.FluentSpeechCommands ---")
    # test_torchaudio_fsc("/workspace/stm/s4/data")