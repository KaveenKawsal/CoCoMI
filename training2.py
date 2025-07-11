import os
# Disable Albumentations update check
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import logging
import numpy as np
from PIL import Image
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
import torch.nn.functional as F
import copy
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.metrics import classification_report
import random  # Add this import


class DoubleConv(nn.Module):
    """[Conv2d => BatchNorm => ReLU] * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Modified filter sizes for 256x256 input
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Update size validation
        if x.size(2) != 256 or x.size(3) != 256:
            raise ValueError("Input image size must be 256x256")
            
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ... (UNet, Down, Up, OutConv, DoubleConv remain unchanged) ...

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.size()

        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)

        value = self.value(x).view(batch, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)

        out = self.gamma * out + x
        return out

class AttentionUNet(UNet):
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet, self).__init__(n_channels, n_classes)
        # Adjust attention blocks for the modified architecture
        self.attention1 = AttentionBlock(64)
        self.attention2 = AttentionBlock(128)
        self.attention3 = AttentionBlock(256)

    def forward(self, x):
        # Update size validation to match new dimensions
        if x.size(2) != 256 or x.size(3) != 256:
            raise ValueError("Input image size must be 256x256")
            
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.attention1(x2)  # Apply attention
        x3 = self.down2(x2)
        x3 = self.attention2(x3)  # Apply attention
        x4 = self.down3(x3)
        x4 = self.attention3(x4)  # Apply attention
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# Add after the AttentionUNet class

class VanillaUNet(nn.Module):
    """Standard UNet without attention mechanisms"""
    def __init__(self, n_channels=3, n_classes=5, bilinear=False):
        super(VanillaUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Modified filter sizes for 256x256 input (same as your UNet)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            raise ValueError("Input image size must be 256x256")
            
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Add individual loss functions after your existing loss classes
class SimpleDiceLoss(nn.Module):
    """Standard Dice Loss without modifications"""
    def __init__(self, smooth=1e-6):
        super(SimpleDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class SimpleFocalLoss(nn.Module):
    """Standard Focal Loss without class weighting"""
    def __init__(self, alpha=1, gamma=2):
        super(SimpleFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CrossEntropyLoss(nn.Module):
    """Standard Cross Entropy Loss"""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.ce(inputs, targets)

class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice

class AdvancedFocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2, smooth=1e-6, alpha=None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_transforms():
    return A.Compose([
        A.Resize(256, 256, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # Removed aggressive augmentations
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])



class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self._validate_directories()
        self.transform = setup_transforms()
        
        self.label_map = {
            0: 0,      # Background
            255: 1,    # Bud-Rot-
            100: 2,    # Dried-Fond-Falling
            150: 3,    # grey leaf spot
            200: 4     # Stem-bleeding
        }
        
        self.class_names = {
            0: "Background",
            1: "Bud-Rot-",
            2: "Dried-Fond-Falling",
            3: "grey leaf spot",
            4: "Stem-bleeding"
        }

    def _validate_directories(self):
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

        image_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.images_dir)}
        mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.masks_dir)}
        common_files = sorted(set(image_files.keys()) & set(mask_files.keys()))

        if len(common_files) == 0:
            raise ValueError("No matching image-mask pairs found!")

        self.images = [image_files[f] for f in common_files]
        self.masks = [mask_files[f] for f in common_files]

    def convert_mask_to_classes(self, mask):
        output_mask = np.zeros_like(mask, dtype=np.int64)
        
        for pixel_value, class_idx in self.label_map.items():
            output_mask[mask == pixel_value] = class_idx
            
        return output_mask

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.images_dir, self.images[idx])
            mask_path = os.path.join(self.masks_dir, self.masks[idx])

            # Load image and mask
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Validate image and mask sizes
            mask = self.convert_mask_to_classes(mask)
            transformed = self.transform(image=image, mask=mask)

            return transformed['image'], transformed['mask'].long()

        except Exception as e:
            logging.error(f"Error loading pair: {self.images[idx]} - {self.masks[idx]}")
            logging.error(f"Error details: {str(e)}")
            raise

    def __len__(self):
        return len(self.images)
    
    def get_num_classes(self):
        return len(self.label_map)
   # ... (SegmentationDataset remains unchanged) ...


def calculate_advanced_class_weights(train_loader, num_classes=5, device='cuda'):
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    for _, masks in train_loader:
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum().item()
        total_pixels += masks.numel()
    
    # Median Frequency Balancing
    freq = class_counts / total_pixels
    median_freq = torch.median(freq)
    class_weights = median_freq / freq
    
    # Normalize and smooth
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.pow(class_weights, 0.5)  # Smooth out extreme weights
    
    return class_weights.to(device)



def setup_data_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,  # Use the provided batch size
        shuffle=True,
        num_workers=2,  # Adjust based on CPU cores
        pin_memory=True
        # Removed persistent_workers and prefetch_factor to reduce memory usage
    )

def dice_loss(pred, target, smooth=1e-5):
    """Calculate Dice Loss."""
    pred = F.softmax(pred, dim=1)  # Apply softmax to get probabilities
    num_classes = pred.shape[1]  # Get number of classes
    dice = torch.zeros(num_classes, device=pred.device)
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice[c] = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
    dice_loss = 1 - dice.mean()  # Average over all classes
    return dice_loss



class DiceLoss(nn.Module):  # As a callable class for easier integration
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        return dice_loss(pred, target, self.smooth)


class FocalLoss(nn.Module):  # Focal Loss implementation
    def __init__(self, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N*H*W,C
        else:
            inputs = inputs.view(-1, inputs.size(1))  # N,C

        targets = targets.view(-1, 1)  # Ensure targets have the correct shape for gather

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)  # Gather the log probabilities corresponding to targets
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.type():
                self.alpha = self.alpha.to(inputs.device).type_as(inputs)
            at = self.alpha.gather(0, targets.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



def get_optimizer(optimizer_name, model_params, learning_rate, weight_decay=1e-4, momentum=0.9):
    """
    Factory function to create optimizers
    """
    optimizers = {
        'sgd': lambda: torch.optim.SGD(
            model_params, 
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        ),
        'adam': lambda: torch.optim.Adam(
            model_params,
            lr=learning_rate,
            weight_decay=weight_decay
        ),
        'adamw': lambda: torch.optim.AdamW(
            model_params,
            lr=learning_rate,
            weight_decay=weight_decay
        ),
        'rmsprop': lambda: torch.optim.RMSprop(
            model_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        ),
        'adadelta': lambda: torch.optim.Adadelta(
            model_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    }
    
    return optimizers.get(optimizer_name.lower())()

def get_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Factory function to create learning rate schedulers
    """
    schedulers = {
        'step': lambda: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 20),
            gamma=kwargs.get('gamma', 0.1)
        ),
        'plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-6)
        ),
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 10),
            eta_min=kwargs.get('min_lr', 0)
        )
    }
    
    return schedulers.get(scheduler_name.lower())()

# Replace your existing setup_model function

def setup_model(device, train_dataset, n_classes=5, learning_rate=0.01, 
                optimizer_name='sgd', scheduler_name='step', 
                model_type='attention_unet', loss_type='advanced_focal_dice'):
    """Setup model with different architectures and loss functions"""
    
    # Model selection
    if model_type == 'vanilla_unet':
        model = VanillaUNet(n_channels=3, n_classes=n_classes)
        print(f"Using Vanilla UNet")
    elif model_type == 'attention_unet':
        model = AttentionUNet(n_channels=3, n_classes=n_classes)
        print(f"Using Attention UNet")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Loss function selection
    if loss_type == 'cross_entropy':
        criterion = CrossEntropyLoss()
    elif loss_type == 'dice':
        criterion = SimpleDiceLoss()
    elif loss_type == 'focal':
        criterion = SimpleFocalLoss()
    elif loss_type == 'advanced_focal_dice':
        # For advanced loss, calculate class weights
        train_loader = setup_data_loader(train_dataset, batch_size=4)
        class_weights = calculate_advanced_class_weights(train_loader, n_classes, device)
        criterion = AdvancedFocalDiceLoss(
            focal_weight=0.3,
            dice_weight=0.7,
            gamma=2,
            alpha=class_weights
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    print(f"Using {loss_type} loss")
    
    # Optimizer
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    
    # Scheduler
    scheduler = get_scheduler(scheduler_name, optimizer)
    
    return model, criterion, optimizer, scheduler

def save_model(model, optimizer, num_epochs, metrics, save_path='unet_model.pth'):
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics['train_losses'],
        'val_loss': metrics['val_losses'],
        'train_acc': metrics['train_accuracies'],
        'val_acc': metrics['val_accuracies'],
        'train_f1': metrics['train_f1s'],
        'val_f1': metrics['val_f1s']
    }, save_path)


def create_datasets(root_dir):
    datasets = []
    disease_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    if not disease_dirs:
        raise ValueError(f"No disease directories found in {root_dir}")
    
    logging.info(f"Found disease directories: {disease_dirs}")
    
    for disease in disease_dirs:
        disease_path = os.path.join(root_dir, disease)
        images_dir = os.path.join(disease_path, 'images')
        masks_dir = os.path.join(disease_path, 'Masks')
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            try:
                dataset = SegmentationDataset(images_dir, masks_dir)
                datasets.append(dataset)
                logging.info(f"Added dataset for {disease}: {len(dataset)} images")
            except Exception as e:
                logging.error(f"Error loading dataset for {disease}: {str(e)}")
        else:
            logging.warning(f"Skipping {disease}: missing images or masks directory")
    
    if not datasets:
        raise ValueError("No valid datasets found!")
        
    class_names = list(datasets[0].class_names.values())
    combined_dataset = ConcatDataset(datasets)
    
    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size   # 20% for validation
    
    # Updated split to only create train and validation sets
    train_dataset, val_dataset = random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logging.info(f"Dataset split sizes:")
    logging.info(f"Total: {total_size}")
    logging.info(f"Train: {train_size}")
    logging.info(f"Validation: {val_size}")
    
    return train_dataset, val_dataset, class_names


def evaluate_model(model, dataloader, criterion, device, class_names):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    # Removed all_probabilities to save memory

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

            # Clear cache
            del images, masks, outputs, probabilities
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=1)
    jaccard = jaccard_score(all_targets, all_predictions, average='weighted', zero_division=1)
    avg_loss = total_loss / len(dataloader)
    
    # Generate classification report
    report = classification_report(
        all_targets, all_predictions, 
        target_names=class_names, 
        output_dict=True,
        zero_division=1
    )

    return avg_loss, accuracy, f1, jaccard, report


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")

    fig.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curves(fpr, tpr, roc_auc, class_names):
    plt.figure()
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()


def plot_metrics(report, epochs):
    metrics = ['precision', 'recall', 'f1-score']
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for cls in classes:
            plt.plot(range(1, epochs + 1), [report_epoch[cls][metric] for report_epoch in report], label=cls)
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} vs. Epochs")
        plt.legend()
        plt.savefig(f"{metric}_plot.png")
        plt.close()


    
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, class_names, patience=15):
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    metrics = {
        'train_losses': [], 'val_losses': [],
        'train_accuracies': [], 'val_accuracies': [],
        'train_f1s': [], 'val_f1s': [],
        'train_jaccards': [], 'val_jaccards': [],
        'val_reports': []
    }

    for epoch in range(1, num_epochs + 1):
        # Training phase
        train_loss, train_acc, train_f1, train_jaccard = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase - Fix: Unpack only 5 values as returned by evaluate_model
        val_loss, val_acc, val_f1, val_jaccard, val_report = evaluate_model(
            model, val_loader, criterion, device, class_names
        )
        
        # Store metrics
        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['train_accuracies'].append(train_acc)
        metrics['val_accuracies'].append(val_acc)
        metrics['train_f1s'].append(train_f1)
        metrics['val_f1s'].append(val_f1)
        metrics['train_jaccards'].append(train_jaccard)
        metrics['val_jaccards'].append(val_jaccard)
        metrics['val_reports'].append(val_report)

        # Log progress
        logging.info(f'Epoch {epoch}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logging.info(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch} epochs')
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy().flatten())
            targets.extend(masks.cpu().numpy().flatten())
            
        # Selective cache clearing
        del outputs, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')  # Changed from weighted
    jaccard = jaccard_score(targets, predictions, average='macro')  # Changed from weighted
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, f1, jaccard

def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, train_jaccards, val_jaccards):  # Updated plotting function
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    # ... (rest of loss plot unchanged)


    # Accuracy
    axes[0, 1].plot(train_accuracies, label='Train Accuracy')
    axes[0, 1].plot(val_accuracies, label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # F1-score
    axes[1, 0].plot(val_f1s, label='Validation F1-score')
    axes[1, 0].set_title('F1-score Over Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-score')
    axes[1, 0].legend()


    # Jaccard Index
    axes[1, 1].plot(val_jaccards, label='Validation Jaccard Index')
    axes[1, 1].set_title('Jaccard Index Over Time')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Jaccard Index')
    axes[1, 1].legend()


    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def train_with_multiple_optimizers(device, train_dataset, val_dataset, class_names, args):
    """Train the model with different optimizers and compare results"""
    optimizers = ['sgd', 'adam', 'adamw', 'rmsprop', 'adadelta']
    results = {}
    
    for opt_name in optimizers:
        logging.info(f"\n{'='*50}")
        logging.info(f"Training with {opt_name.upper()} optimizer")
        logging.info(f"{'='*50}\n")
        
        # Setup model, criterion, optimizer and scheduler
        model, criterion, optimizer, scheduler = setup_model(
            device, 
            train_dataset, 
            learning_rate=args.learning_rate,
            optimizer_name=opt_name,
            scheduler_name=args.scheduler
        )
        
        train_loader = setup_data_loader(train_dataset, args.batch_size)
        val_loader = setup_data_loader(val_dataset, args.batch_size)
        
        # Train model
        model, metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            scheduler, device, args.num_epochs, class_names
        )
        
        # Store results
        results[opt_name] = {
            'final_train_loss': metrics['train_losses'][-1],
            'final_val_loss': metrics['val_losses'][-1],
            'final_val_accuracy': metrics['val_accuracies'][-1],
            'final_val_f1': metrics['val_f1s'][-1],
            'final_val_jaccard': metrics['val_jaccards'][-1],
            'train_losses': metrics['train_losses'],
            'val_losses': metrics['val_losses'],
            'val_accuracies': metrics['val_accuracies'],
            'val_f1s': metrics['val_f1s']
        }
        
        # Save model for this optimizer
        SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f'model_{opt_name}.pth')
        save_model(model, optimizer, args.num_epochs, metrics, save_path=save_path)
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
    
    return results

# Add this function after train_with_multiple_optimizers

def train_ablation_study(device, train_dataset, val_dataset, class_names, args):
    """Perform ablation study with different models and loss functions"""
    
    # Define variants to test
    variants = [
        {'model_type': 'vanilla_unet', 'loss_type': 'cross_entropy', 'name': 'Vanilla_UNet_CrossEntropy'},
        {'model_type': 'vanilla_unet', 'loss_type': 'dice', 'name': 'Vanilla_UNet_Dice'},
        {'model_type': 'vanilla_unet', 'loss_type': 'focal', 'name': 'Vanilla_UNet_Focal'},
        {'model_type': 'attention_unet', 'loss_type': 'cross_entropy', 'name': 'Attention_UNet_CrossEntropy'},
        {'model_type': 'attention_unet', 'loss_type': 'dice', 'name': 'Attention_UNet_Dice'},
        {'model_type': 'attention_unet', 'loss_type': 'focal', 'name': 'Attention_UNet_Focal'},
        {'model_type': 'attention_unet', 'loss_type': 'advanced_focal_dice', 'name': 'Attention_UNet_AdvancedFocalDice'},
    ]
    
    results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Training: {variant['name']}")
        print(f"{'='*60}")
        
        # Setup data loaders
        train_loader = setup_data_loader(train_dataset, args.batch_size)
        val_loader = setup_data_loader(val_dataset, args.batch_size)
        
        # Setup model with specific variant
        model, criterion, optimizer, scheduler = setup_model(
            device, train_dataset, n_classes=len(class_names),
            learning_rate=args.learning_rate, optimizer_name='adam',  # Use Adam for consistency
            scheduler_name=args.scheduler, model_type=variant['model_type'],
            loss_type=variant['loss_type']
        )
        
        # Train model
        model, metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            scheduler, device, args.num_epochs, class_names
        )
        
        # Store results
        results[variant['name']] = {
            'final_train_loss': metrics['train_losses'][-1],
            'final_val_loss': metrics['val_losses'][-1],
            'final_val_accuracy': metrics['val_accuracies'][-1],
            'final_val_f1': metrics['val_f1s'][-1],
            'final_val_jaccard': metrics['val_jaccards'][-1],
            'train_losses': metrics['train_losses'],
            'val_losses': metrics['val_losses'],
            'val_accuracies': metrics['val_accuracies'],
            'val_f1s': metrics['val_f1s'],
            'val_jaccards': metrics['val_jaccards'],
            'model_type': variant['model_type'],
            'loss_type': variant['loss_type']
        }
        
        # Save model for this variant
        SAVE_DIR = os.path.join(os.getcwd(), 'saved_models', 'ablation_study')
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f'{variant["name"]}.pth')
        save_model(model, optimizer, args.num_epochs, metrics, save_path=save_path)
        
        # Clear memory
        del model, optimizer, scheduler, criterion, train_loader, val_loader
        torch.cuda.empty_cache()
    
    return results

def plot_optimizer_comparison(results):
    """Plot comparison of different optimizers"""
    metrics = ['train_losses', 'val_losses', 'val_accuracies', 'val_f1s']
    titles = ['Training Loss', 'Validation Loss', 'Validation Accuracy', 'Validation F1 Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        for opt_name, opt_results in results.items():
            axes[idx].plot(opt_results[metric], label=opt_name.upper())
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Epoch')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    plt.close()

# Add this function after plot_optimizer_comparison

def plot_ablation_comparison(results):
    """Plot comparison of different model architectures and loss functions"""
    
    # Separate results by model type
    vanilla_results = {k: v for k, v in results.items() if 'Vanilla' in k}
    attention_results = {k: v for k, v in results.items() if 'Attention' in k}
    
    # Plot 1: Model Architecture Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics = ['val_losses', 'val_accuracies', 'val_f1s', 'val_jaccards']
    titles = ['Validation Loss', 'Validation Accuracy', 'Validation F1 Score', 'Validation Jaccard Index']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        # Plot vanilla UNet variants
        for name, result in vanilla_results.items():
            loss_name = name.split('_')[-1]
            axes[idx].plot(result[metric], label=f'Vanilla UNet + {loss_name}', linestyle='--')
        
        # Plot attention UNet variants
        for name, result in attention_results.items():
            loss_name = name.split('_')[-1]
            axes[idx].plot(result[metric], label=f'Attention UNet + {loss_name}')
        
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Epoch')
        axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('ablation_study_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 2: Final Results Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    names = list(results.keys())
    final_f1s = [results[name]['final_val_f1'] for name in names]
    final_accuracies = [results[name]['final_val_accuracy'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_f1s, width, label='F1 Score', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_accuracies, width, label='Accuracy', alpha=0.8)
    
    ax.set_xlabel('Model Variants')
    ax.set_ylabel('Score')
    ax.set_title('Final Validation Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ') for name in names], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ablation_final_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    set_seed()
    setup_logging()
    parser = argparse.ArgumentParser(description='Coconut Tree Disease Detection')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--scheduler', type=str, default='step',
                      choices=['step', 'plateau', 'cosine'],
                      help='Learning rate scheduler to use')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()

    device = setup_device()

    if args.train or args.ablation:
        try:
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            
            root_dir = r"S:\SEM3\COMPUTER NETWORKS\project\train"
            train_dataset, val_dataset, class_names = create_datasets(root_dir)
            
            if args.ablation:
                # Run ablation study
                print("Starting Ablation Study...")
                results = train_ablation_study(
                    device, train_dataset, val_dataset, class_names, args
                )
                
                # Print ablation results
                print("\nAblation Study Results:")
                print("=" * 80)
                metrics_to_compare = ['final_val_loss', 'final_val_accuracy', 'final_val_f1', 'final_val_jaccard']
                
                for metric in metrics_to_compare:
                    print(f"\n{metric}:")
                    for variant_name, variant_results in results.items():
                        print(f"{variant_name}: {variant_results[metric]:.4f}")
                
                # Plot ablation comparison
                plot_ablation_comparison(results)
                
                # Save ablation results to file
                with open('ablation_study_results.txt', 'w') as f:
                    f.write("Ablation Study Results\n")
                    f.write("=" * 50 + "\n\n")
                    for variant_name, variant_results in results.items():
                        f.write(f"\nResults for {variant_name}:\n")
                        f.write(f"Model Type: {variant_results['model_type']}\n")
                        f.write(f"Loss Type: {variant_results['loss_type']}\n")
                        for metric, value in variant_results.items():
                            if not isinstance(value, (list, str)):
                                f.write(f"{metric}: {value:.4f}\n")
                        f.write("-" * 40 + "\n")
            
            elif args.train:
                # Train with all optimizers (original functionality)
                results = train_with_multiple_optimizers(
                    device, train_dataset, val_dataset, class_names, args
                )
                
                # Print comparison
                print("\nOptimizer Comparison Results:")
                print("-" * 50)
                metrics_to_compare = ['final_val_loss', 'final_val_accuracy', 'final_val_f1', 'final_val_jaccard']
                
                for metric in metrics_to_compare:
                    print(f"\n{metric}:")
                    for opt_name, opt_results in results.items():
                        print(f"{opt_name.upper()}: {opt_results[metric]:.4f}")
                
                # Plot comparison
                plot_optimizer_comparison(results)
                
                # Save results to file
                with open('optimizer_comparison_results.txt', 'w') as f:
                    for opt_name, opt_results in results.items():
                        f.write(f"\nResults for {opt_name.upper()}:\n")
                        for metric, value in opt_results.items():
                            if not isinstance(value, list):
                                f.write(f"{metric}: {value:.4f}\n")

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    main()
