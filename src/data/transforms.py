import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
import math


class RandomRotation(transforms.RandomRotation):
    """Random rotation with better default interpolation."""
    
    def __init__(self, degrees, interpolation=Image.BILINEAR, expand=False, center=None, fill=0):
        super().__init__(degrees, interpolation, expand, center, fill)


class TwinFaceTransforms:
    """Transforms for twin face dataset."""
    
    @staticmethod
    def get_train_transforms(config):
        """Get training transforms."""
        augmentation = config['data']['augmentation']
        
        transforms_list = [
            transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
        ]
        
        # Add augmentations
        if augmentation.get('horizontal_flip', 0) > 0:
            transforms_list.append(
                transforms.RandomHorizontalFlip(p=augmentation['horizontal_flip'])
            )
        
        if augmentation.get('rotation', 0) > 0:
            transforms_list.append(
                RandomRotation(degrees=augmentation['rotation'])
            )
        
        if augmentation.get('color_jitter'):
            cj = augmentation['color_jitter']
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                )
            )
        
        # Convert to tensor and normalize
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Random erasing (after normalization)
        if augmentation.get('random_erasing', 0) > 0:
            transforms_list.append(
                transforms.RandomErasing(p=augmentation['random_erasing'])
            )
        
        return transforms.Compose(transforms_list)
    
    @staticmethod
    def get_val_transforms(config):
        """Get validation transforms."""
        return transforms.Compose([
            transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_test_transforms(config):
        """Get test transforms (same as validation)."""
        return TwinFaceTransforms.get_val_transforms(config)


class TenCrop:
    """Ten crop augmentation for test time."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        return transforms.functional.ten_crop(img, self.size)


class FiveCrop:
    """Five crop augmentation for test time."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        return transforms.functional.five_crop(img, self.size)
