import os
import argparse
import fiftyone as fo
import fiftyone.zoo as foz

def download_coco_fiftyone(split="validation", max_samples=None, data_root="data"):
    """Download COCO with FiftyOne (more Pythonic)"""
    print(f"Downloading COCO 2017 {split}...")
    
    # FiftyOne manages its own dataset location by default, but we can export it or just use it.
    # Here we load it into FiftyOne's database.
    
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        max_samples=max_samples,
        dataset_name=f"coco-2017-{split}"
    )
    
    print(f"✓ COCO {split}: {len(dataset)} samples")
    
    # If we want to export it to a standard directory structure:
    # export_dir = os.path.join(data_root, "coco-2017", split)
    # dataset.export(
    #     export_dir=export_dir,
    #     dataset_type=fo.types.COCODetectionDataset,
    #     label_field="ground_truth",
    # )
    # print(f"✓ Exported to {export_dir}")
    
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download COCO 2017 dataset')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'validation', 'test'],
                       help='Split to download')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to download')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Root directory for datasets (optional export)')
    
    args = parser.parse_args()
    
    download_coco_fiftyone(split=args.split, max_samples=args.max_samples, data_root=args.data_root)
