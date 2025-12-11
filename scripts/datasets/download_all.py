import argparse
import sys
import os

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_ade20k import download_ade20k
from download_coco import download_coco_fiftyone
from download_voc import download_voc_torchvision

def main():
    parser = argparse.ArgumentParser(description='Download evaluation datasets')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['ade20k', 'coco', 'voc', 'all'],
                       default=['all'],
                       help='Which datasets to download')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Root directory for datasets')
    parser.add_argument('--quick', action='store_true',
                       help='Download small subsets for testing (where applicable)')
    
    args = parser.parse_args()
    
    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['ade20k', 'coco', 'voc']
    
    print(f"📥 Downloading: {datasets_to_download}")
    
    os.makedirs(args.data_root, exist_ok=True)
    
    if 'ade20k' in datasets_to_download:
        print("\n[1/3] ADE20K...")
        download_ade20k(f"{args.data_root}/ade20k")
    
    if 'coco' in datasets_to_download:
        print("\n[2/3] COCO 2017...")
        max_samples = 100 if args.quick else None
        download_coco_fiftyone(split="validation", max_samples=max_samples, data_root=args.data_root)
    
    if 'voc' in datasets_to_download:
        print("\n[3/3] PASCAL VOC 2012...")
        download_voc_torchvision(args.data_root)
    
    print("\n✅ All requested datasets processed!")

if __name__ == "__main__":
    main()
