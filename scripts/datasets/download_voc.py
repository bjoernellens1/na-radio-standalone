import os
import argparse
from torchvision.datasets import VOCSegmentation

def download_voc_torchvision(root="data", year='2012', image_set='val'):
    print(f"Downloading PASCAL VOC {year} {image_set}...")
    
    # Ensure root exists
    os.makedirs(root, exist_ok=True)
    
    dataset = VOCSegmentation(
        root=root,
        year=year,
        image_set=image_set,
        download=True
    )
    print(f"✓ VOC {year} {image_set} via torchvision: {len(dataset)} samples")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download PASCAL VOC dataset')
    parser.add_argument('--root', type=str, default='data',
                       help='Root directory for datasets')
    parser.add_argument('--year', type=str, default='2012',
                       help='Year of the dataset')
    parser.add_argument('--image-set', type=str, default='val', choices=['train', 'trainval', 'val'],
                       help='Image set to download')
    
    args = parser.parse_args()
    
    download_voc_torchvision(root=args.root, year=args.year, image_set=args.image_set)
