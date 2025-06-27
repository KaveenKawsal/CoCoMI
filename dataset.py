# dataset.py

import os
import cv2
import numpy as np
import logging
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def setup_transforms():
    """Defines the image augmentation pipeline."""
    return A.Compose([
        A.Resize(256, 256, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class SegmentationDataset(Dataset):
    """Custom Dataset for semantic segmentation."""
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform if transform else setup_transforms()
        
        self.label_map = {0: 0, 255: 1, 100: 2, 150: 3, 200: 4}
        self.class_names = {0: "Background", 1: "Bud-Rot-", 2: "Dried-Fond-Falling", 3: "grey leaf spot", 4: "Stem-bleeding"}
        
        self._validate_and_load_files()

    def _validate_and_load_files(self):
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        image_files = {os.path.splitext(f)[0] for f in os.listdir(self.images_dir)}
        mask_files = {os.path.splitext(f)[0] for f in os.listdir(self.masks_dir)}
        
        # Find common files and assume they have common extensions like .jpg, .png
        common_basenames = sorted(list(image_files & mask_files))
        
        if not common_basenames:
            raise ValueError("No matching image-mask pairs found!")

        # Reconstruct full filenames (this is more robust to different extensions)
        all_image_files = os.listdir(self.images_dir)
        all_mask_files = os.listdir(self.masks_dir)
        
        self.images = [f for f in all_image_files if os.path.splitext(f)[0] in common_basenames]
        self.masks = [f for f in all_mask_files if os.path.splitext(f)[0] in common_basenames]

        logging.info(f"Found {len(self.images)} matching pairs in {os.path.basename(self.images_dir)}.")


    def _convert_mask_to_classes(self, mask):
        output_mask = np.zeros_like(mask, dtype=np.int64)
        for pixel_value, class_idx in self.label_map.items():
            output_mask[mask == pixel_value] = class_idx
        return output_mask

    def __len__(self):
        return len(self.images)
    
    def get_num_classes(self):
        return len(self.label_map)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        mask = self._convert_mask_to_classes(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.long()

def create_datasets(root_dir, val_split=0.2):
    """Creates and splits datasets from a root directory."""
    datasets = []
    disease_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    if not disease_dirs:
        raise ValueError(f"No disease subdirectories found in {root_dir}")
    
    logging.info(f"Found disease directories: {disease_dirs}")
    
    for disease in disease_dirs:
        disease_path = os.path.join(root_dir, disease)
        images_dir = os.path.join(disease_path, 'images')
        masks_dir = os.path.join(disease_path, 'Masks')
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            try:
                dataset = SegmentationDataset(images_dir, masks_dir)
                if len(dataset) > 0:
                    datasets.append(dataset)
            except Exception as e:
                logging.warning(f"Could not load dataset for {disease}: {e}")
        else:
            logging.warning(f"Skipping {disease}: missing 'images' or 'Masks' directory.")
    
    if not datasets:
        raise ValueError("No valid datasets could be created.")
        
    class_names = list(datasets[0].class_names.values())
    combined_dataset = ConcatDataset(datasets)
    
    total_size = len(combined_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logging.info(f"Total samples: {total_size}, Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    return train_dataset, val_dataset, class_names

def setup_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True if torch.cuda.is_available() else False
    )