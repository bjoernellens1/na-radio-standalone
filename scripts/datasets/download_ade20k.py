from datasets import load_dataset
import os
import argparse

def download_ade20k(save_dir="data/ade20k"):
    """Download ADE20K via HuggingFace datasets"""
    print("Downloading ADE20K (Scene Parse 150)...")
    
    # Load full dataset
    try:
        dataset = load_dataset("scene_parse_150", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load without trust_remote_code=True (might fail if script is required)...")
        dataset = load_dataset("scene_parse_150")
    
    # Save locally
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)
    print(f"✓ ADE20K saved to {save_dir}")
    
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ADE20K dataset')
    parser.add_argument('--save-dir', type=str, default='data/ade20k',
                       help='Directory to save the dataset')
    args = parser.parse_args()
    
    dataset = download_ade20k(save_dir=args.save_dir)
    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")
