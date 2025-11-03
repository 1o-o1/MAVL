import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def preprocess_image(
    image: Image.Image,
    size: int = 224,
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    Basic preprocessing for a single image.
    
    Args:
        image: PIL Image
        size: Target size for the image
        normalize_mean: Mean values for normalization
        normalize_std: Std values for normalization
        
    Returns:
        Preprocessed image tensor of shape (3, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    return transform(image)


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.225],
    use_augmentation: bool = True
) -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Args:
        image_size: Target image size
        is_training: Whether to apply training augmentations
        normalize_mean: Mean values for normalization
        normalize_std: Std values for normalization
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Composed transforms
    """
    if is_training and use_augmentation:
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),  # Resize larger for cropping
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ]
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ]
    
    return transforms.Compose(transform_list)


def get_albumentations_transforms(
    image_size: int = 224,
    is_training: bool = True,
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    """
    Get Albumentations transforms for more advanced augmentations.
    
    Args:
        image_size: Target image size
        is_training: Whether to apply training augmentations
        normalize_mean: Mean values for normalization
        normalize_std: Std values for normalization
        
    Returns:
        Albumentations Compose object
    """
    if is_training:
        transform = A.Compose([
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
                A.RandomGamma(gamma_limit=(90, 110), p=1),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
            ], p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.5
            ),
            A.GridDistortion(p=0.2),
            A.ElasticTransform(p=0.2),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2()
        ])
    
    return transform


class AlbumentationsWrapper:
    """Wrapper to use Albumentations with PyTorch datasets."""
    
    def __init__(self, transform: A.Compose):
        self.transform = transform
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Apply albumentations
        augmented = self.transform(image=image_np)
        
        return augmented['image']


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean