import os
import glob
import cv2
import numpy as np
import torch
import urllib.request
import zipfile
import tarfile
import shutil
from sklearn.metrics import jaccard_score, recall_score
from PIL import Image

class DatasetLoader:
    def __init__(self, root_path):
        self.root_path = root_path
        
        # Try standard structure
        self.images_path = os.path.join(root_path, 'images')
        self.masks_path = os.path.join(root_path, 'masks')
        
        if not os.path.exists(self.images_path) or not os.path.exists(self.masks_path):
            # Try VOC structure
            # Check for VOCdevkit/VOC2012 subdir or direct
            potential_images = []
            potential_masks = []
            
            # Walk to find JPEGImages and SegmentationClass
            for root, dirs, files in os.walk(root_path):
                if 'JPEGImages' in dirs:
                    potential_images.append(os.path.join(root, 'JPEGImages'))
                if 'SegmentationClass' in dirs:
                    potential_masks.append(os.path.join(root, 'SegmentationClass'))
            
            if potential_images and potential_masks:
                # Use the first pair found (assuming one dataset per root)
                self.images_path = potential_images[0]
                self.masks_path = potential_masks[0]
                print(f"Detected VOC structure: {self.images_path}, {self.masks_path}")
            else:
                # Try Semantic-NeRF Replica structure
                # Look for 'rgb' and 'semantic_class' folders
                potential_images = []
                potential_masks = []
                for root, dirs, files in os.walk(root_path):
                    if 'rgb' in dirs and 'semantic_class' in dirs:
                        potential_images.append(os.path.join(root, 'rgb'))
                        potential_masks.append(os.path.join(root, 'semantic_class'))
                
                if potential_images and potential_masks:
                    self.images_path = potential_images[0]
                    self.masks_path = potential_masks[0]
                    print(f"Detected Semantic-NeRF Replica structure: {self.images_path}, {self.masks_path}")
                else:
                    # Try Nice-SLAM Replica structure (No masks usually)
                    # Look for 'results' folder with 'frame*.jpg'
                    potential_images = []
                    for root, dirs, files in os.walk(root_path):
                        if 'results' in dirs:
                            # Check if it contains frames
                            if any(f.startswith('frame') and f.endswith('.jpg') for f in os.listdir(os.path.join(root, 'results'))):
                                potential_images.append(os.path.join(root, 'results'))
                    
                    if potential_images:
                        self.images_path = potential_images[0]
                        self.masks_path = None # No masks
                        print(f"Detected Nice-SLAM Replica structure (Images only): {self.images_path}")
                    else:
                         raise ValueError(f"Dataset structure not recognized in {root_path}")
            
        self.image_files = sorted(glob.glob(os.path.join(self.images_path, '*')))
        if self.masks_path:
            self.mask_files = sorted(glob.glob(os.path.join(self.masks_path, '*')))
            # Filter to keep only matching pairs
            # Note: Semantic-NeRF uses rgb_0.png and semantic_class_0.png. 
            # Basenames: rgb_0 vs semantic_class_0. They don't match directly.
            # We need a smarter matching strategy.
            
            # Helper to extract ID
            def get_id(fname):
                # Extract number from filename
                import re
                match = re.search(r'(\d+)', fname)
                return int(match.group(1)) if match else -1
                
            # Create dict mapping ID to filename
            img_map = {get_id(os.path.basename(f)): f for f in self.image_files}
            mask_map = {get_id(os.path.basename(f)): f for f in self.mask_files}
            
            # Find common IDs
            common_ids = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
            
            self.image_files = [img_map[i] for i in common_ids]
            self.mask_files = [mask_map[i] for i in common_ids]
        else:
            self.mask_files = []
        
        # Load class mapping if exists
        self.classes = {}
        class_file = os.path.join(root_path, 'classes.txt')
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        self.classes[int(parts[0])] = parts[1]
        else:
            # Default or try to infer?
            print("No classes.txt found. Using raw pixel values as class IDs.")
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.mask_files:
            mask_path = self.mask_files[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
            
        return img, mask, os.path.basename(img_path)

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate metrics between binary prediction mask and binary ground truth mask.
    pred_mask: boolean or 0/1 numpy array
    gt_mask: boolean or 0/1 numpy array
    """
    # Resize pred_mask to match gt_mask if needed
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    # mIoU (for binary case, it's just IoU of the positive class)
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()
    iou = intersection / (union + 1e-8)
    
    # Recall
    # TP / (TP + FN)
    tp = intersection
    fn = np.logical_and(np.logical_not(pred_flat), gt_flat).sum()
    recall = tp / (tp + fn + 1e-8)
    
    # SCV (Spatial Consistency Value) - Placeholder
    # Let's implement a simple version: 1 - (variance of prediction in GT foreground region)
    # Or maybe boundary alignment?
    # For now, return 0.0
    scv = 0.0
    
    # SCVR - Placeholder
    scvr = 0.0
    
    return {
        'mIoU': float(iou),
        'Recall': float(recall),
        'SCV': scv,
        'SCVR': scvr
    }

class ComparisonEngine:
    def __init__(self):
        self.dataset = None
        self.results = {}
        
    def load_dataset(self, path):
        self.dataset = DatasetLoader(path)
        return len(self.dataset), self.dataset.classes
        
    def download_and_extract_dataset(self, url, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        filename = url.split('/')[-1]
        local_path = os.path.join(save_path, filename)
        
        print(f"Downloading {url} to {local_path}...")
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")
            
        print(f"Extracting {local_path}...")
        try:
            if filename.endswith('.zip'):
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(save_path)
            elif filename.endswith(('.tar.gz', '.tgz', '.tar')):
                with tarfile.open(local_path, 'r') as tar_ref:
                    tar_ref.extractall(save_path)
            else:
                raise ValueError("Unsupported archive format. Use .zip or .tar.gz")
        except Exception as e:
             raise RuntimeError(f"Extraction failed: {e}")
        finally:
            # Cleanup archive?
            # os.remove(local_path)
            pass
            
        # Check if extracted directly or into subdir
        # If single subdir, maybe move contents up?
        # For now, let the user specify the correct path or we return the path found.
        # Let's try to find 'images' folder
        
        found_path = save_path
        for root, dirs, files in os.walk(save_path):
            if 'images' in dirs and 'masks' in dirs:
                found_path = root
                break
                
        return found_path

    def run_comparison(self, encoder, models_to_compare, selected_classes=None):
        # This function might need to be a generator or async to report progress
        pass
