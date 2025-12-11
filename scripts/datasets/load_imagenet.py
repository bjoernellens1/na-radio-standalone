from torchvision.datasets import ImageNet
import os
import argparse

def load_imagenet_val(root="data/imagenet"):
    """Load pre-downloaded ImageNet validation set"""
    # Assumes you've manually downloaded ILSVRC2012_img_val.tar
    # and placed it in root/
    
    print(f"Loading ImageNet from {root}...")
    
    if not os.path.exists(root):
        print(f"Error: {root} does not exist. Please download ImageNet manually.")
        return None

    try:
        dataset = ImageNet(
            root=root,
            split='val'
        )
        print(f"✓ ImageNet validation: {len(dataset)} images")
        return dataset
    except RuntimeError as e:
        print(f"Error loading ImageNet: {e}")
        print("Ensure you have downloaded the validation set archive to the root directory.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load ImageNet dataset')
    parser.add_argument('--root', type=str, default='data/imagenet',
                       help='Root directory containing ImageNet files')
    
    args = parser.parse_args()
    
    load_imagenet_val(root=args.root)
