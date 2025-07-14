import json
import os
from typing import Dict, List, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class TwinFaceDataset(Dataset):
    """Dataset for twin face verification task."""
    
    def __init__(
        self,
        dataset_info_path: str,
        twin_pairs_path: str,
        dataset_root: str,
        transform=None,
        is_training: bool = True
    ):
        """
        Args:
            dataset_info_path: Path to JSON file with person_id -> image_paths mapping
            twin_pairs_path: Path to JSON file with twin pairs
            dataset_root: Root directory for images
            transform: Transform to apply to images
            is_training: Whether this is training set
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.is_training = is_training
        
        # Load dataset information
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Create person_id to index mapping
        self.person_ids = list(self.dataset_info.keys())
        self.person_to_idx = {pid: idx for idx, pid in enumerate(self.person_ids)}
        
        # Create twin pairs mapping for easy lookup
        self.twin_map = {}
        for pair in self.twin_pairs:
            self.twin_map[pair[0]] = pair[1]
            self.twin_map[pair[1]] = pair[0]
        
        # Collect all image paths with labels
        self.samples = []
        for person_id, image_paths in self.dataset_info.items():
            label = self.person_to_idx[person_id]
            for img_path in image_paths:
                self.samples.append((img_path, label, person_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, person_id = self.samples[idx]
        
        # Load image
        full_path = os.path.join(self.dataset_root, os.path.basename(img_path))
        if not os.path.exists(full_path):
            # Try the original path
            full_path = img_path
        
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'person_id': person_id,
            'image_path': img_path
        }
    
    def get_twin_id(self, person_id: str) -> Optional[str]:
        """Get the twin ID for a given person ID."""
        return self.twin_map.get(person_id)
    
    def is_twin_pair(self, person_id1: str, person_id2: str) -> bool:
        """Check if two person IDs form a twin pair."""
        return self.twin_map.get(person_id1) == person_id2


class TripletDataset(Dataset):
    """Dataset for triplet loss training."""
    
    def __init__(
        self,
        base_dataset: TwinFaceDataset,
        samples_per_epoch: int = 10000
    ):
        """
        Args:
            base_dataset: Base TwinFaceDataset
            samples_per_epoch: Number of triplets to generate per epoch
        """
        self.base_dataset = base_dataset
        self.samples_per_epoch = samples_per_epoch
        
        # Group samples by person_id for efficient sampling
        self.person_samples = {}
        for idx, (_, label, person_id) in enumerate(base_dataset.samples):
            if person_id not in self.person_samples:
                self.person_samples[person_id] = []
            self.person_samples[person_id].append(idx)
        
        # Filter persons with at least 2 images
        self.valid_persons = [
            pid for pid, samples in self.person_samples.items() 
            if len(samples) >= 2
        ]
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Sample anchor person
        anchor_person = random.choice(self.valid_persons)
        anchor_samples = self.person_samples[anchor_person]
        
        # Sample anchor and positive from same person
        anchor_idx, positive_idx = random.sample(anchor_samples, 2)
        
        # Sample negative from different person (prefer twin for hard negatives)
        twin_id = self.base_dataset.get_twin_id(anchor_person)
        if twin_id and twin_id in self.person_samples and random.random() < 0.5:
            # Use twin as hard negative
            negative_person = twin_id
        else:
            # Use random different person
            negative_person = random.choice([
                pid for pid in self.valid_persons if pid != anchor_person
            ])
        
        negative_idx = random.choice(self.person_samples[negative_person])
        
        # Get samples
        anchor = self.base_dataset[anchor_idx]
        positive = self.base_dataset[positive_idx]
        negative = self.base_dataset[negative_idx]
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }


class PairDataset(Dataset):
    """Dataset for PWCA pair sampling."""
    
    def __init__(self, base_dataset: TwinFaceDataset):
        """
        Args:
            base_dataset: Base TwinFaceDataset
        """
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get primary sample
        primary = self.base_dataset[idx]
        
        # Sample a second image randomly
        secondary_idx = random.randint(0, len(self.base_dataset) - 1)
        secondary = self.base_dataset[secondary_idx]
        
        return {
            'primary': primary,
            'secondary': secondary
        }


def create_data_loaders(
    config: Dict,
    train_transform=None,
    val_transform=None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    # Create datasets
    train_dataset = TwinFaceDataset(
        dataset_info_path=config['data']['train_info'],
        twin_pairs_path=config['data']['train_pairs'],
        dataset_root=config['data']['dataset_root'],
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = TwinFaceDataset(
        dataset_info_path=config['data']['test_info'],
        twin_pairs_path=config['data']['test_pairs'],
        dataset_root=config['data']['dataset_root'],
        transform=val_transform,
        is_training=False
    )
    
    # Create triplet datasets for training
    triplet_train_dataset = TripletDataset(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        triplet_train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader


def create_pair_loader(
    config: Dict,
    base_dataset: TwinFaceDataset
) -> DataLoader:
    """Create data loader for PWCA pair sampling."""
    
    pair_dataset = PairDataset(base_dataset)
    
    return DataLoader(
        pair_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
