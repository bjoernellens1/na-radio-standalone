import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation, ImageNet
from datasets import load_dataset as load_hf_dataset
import fiftyone.zoo as foz
import os
from PIL import Image

def get_transforms(resolution=512):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_ade20k_loader(resolution=512, batch_size=8, root="data/ade20k"):
    # ADE20K via HuggingFace datasets
    # Note: HF datasets handles loading differently, we might need a custom wrapper
    # to make it work seamlessly with PyTorch DataLoader if not using their format
    
    try:
        dataset = load_hf_dataset("scene_parse_150", split="validation", cache_dir=root, trust_remote_code=True)
    except:
         dataset = load_hf_dataset("scene_parse_150", split="validation", cache_dir=root)

    transform = get_transforms(resolution)
    
    def collate_fn(batch):
        images = [transform(item['image'].convert("RGB")) for item in batch]
        masks = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST)(
                    transforms.CenterCrop(resolution)(transforms.ToTensor()(item['annotation']))
                 ) for item in batch] # Simplified mask processing
        return torch.stack(images), torch.stack(masks)

    # Note: This is a simplified loader. For real training/eval, we need proper mask handling.
    # HF dataset returns PIL images.
    
    # For now, let's wrap it in a simple class if needed, or just use the HF dataset directly
    # But HF dataset doesn't support transforms in the same way as torchvision
    
    class ADE20KWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.dataset = hf_dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image'].convert("RGB")
            # Mask handling is complex for ADE20K (150 classes), simplified here
            mask = item['annotation'] 
            
            if self.transform:
                image = self.transform(image)
                
            return image, mask # Returning raw mask for now

    wrapper = ADE20KWrapper(dataset, transform)
    return DataLoader(wrapper, batch_size=batch_size, shuffle=False, num_workers=4)

def build_voc_loader(resolution=512, batch_size=8, root="data"):
    transform = get_transforms(resolution)
    target_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(), # This scales to [0,1], might need to scale back to 0-255 for classes
    ])
    
    dataset = VOCSegmentation(root=root, year='2012', image_set='val', download=False, 
                              transform=transform, target_transform=target_transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def build_coco_loader(resolution=512, batch_size=8, root="data"):
    # COCO via FiftyOne or generic ImageFolder if exported
    # For simplicity, let's assume we want to load it for zero-shot classification or similar
    # If using FiftyOne, it's best to use their torch loader
    
    # Placeholder for COCO loader
    print("COCO loader not fully implemented yet, requires FiftyOne dataset export.")
    return None

def build_imagenet_loader(resolution=512, batch_size=32, root="data/imagenet"):
    transform = get_transforms(resolution)
    dataset = ImageNet(root=root, split='val', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

def get_dataloader(dataset_name, resolution=512, batch_size=8, data_root="data"):
    if dataset_name == "ade20k":
        return build_ade20k_loader(resolution, batch_size, root=f"{data_root}/ade20k")
    elif dataset_name == "voc2012":
        return build_voc_loader(resolution, batch_size, root=data_root)
    elif dataset_name == "coco":
        return build_coco_loader(resolution, batch_size, root=data_root)
    elif dataset_name == "imagenet":
        return build_imagenet_loader(resolution, batch_size, root=f"{data_root}/imagenet")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
