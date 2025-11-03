import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import List, Optional, Callable


def get_augmentation_pipeline(
    image_size: int = 224,
    augmentation_level: str = 'medium',
    use_albumentations: bool = True,
    p_augment: float = 0.5
) -> Callable:
    """
    Get augmentation pipeline based on specified level.
    
    Args:
        image_size: Target image size
        augmentation_level: Level of augmentation ('light', 'medium', 'heavy')
        use_albumentations: Whether to use Albumentations library
        p_augment: Overall probability of applying augmentations
        
    Returns:
        Augmentation pipeline
    """
    if use_albumentations:
        return get_albumentations_pipeline(image_size, augmentation_level, p_augment)
    else:
        return get_torchvision_pipeline(image_size, augmentation_level)


def get_albumentations_pipeline(
    image_size: int = 224,
    augmentation_level: str = 'medium',
    p_augment: float = 0.5
) -> A.Compose:
    """
    Get Albumentations augmentation pipeline.
    
    Args:
        image_size: Target image size
        augmentation_level: Level of augmentation
        p_augment: Overall probability
        
    Returns:
        Albumentations Compose object
    """
    # Base transforms (always applied)
    base_transforms = [
        A.Resize(image_size + 32, image_size + 32),
        A.RandomCrop(image_size, image_size),
    ]
    
    if augmentation_level == 'light':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ]
    
    elif augmentation_level == 'medium':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ], p=0.3),
        ]
    
    elif augmentation_level == 'heavy':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=30,
                p=0.7
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1
                ),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0), p=1),
                A.GaussianBlur(blur_limit=5, p=1),
                A.MedianBlur(blur_limit=5, p=1),
                A.MotionBlur(blur_limit=5, p=1),
            ], p=0.5),
            A.OneOf([
                A.GridDistortion(p=1),
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.3
            ),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    # Normalization (always applied)
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Combine all transforms
    all_transforms = base_transforms + aug_transforms + [normalize, ToTensorV2()]
    
    return A.Compose(all_transforms, p=p_augment)


def get_torchvision_pipeline(
    image_size: int = 224,
    augmentation_level: str = 'medium'
) -> transforms.Compose:
    """
    Get torchvision augmentation pipeline.
    
    Args:
        image_size: Target image size
        augmentation_level: Level of augmentation
        
    Returns:
        Torchvision Compose object
    """
    if augmentation_level == 'light':
        transform_list = [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
        ]
    
    elif augmentation_level == 'medium':
        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ]
    
    elif augmentation_level == 'heavy':
        transform_list = [
            transforms.Resize((image_size + 48, image_size + 48)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15
            ),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    # Add normalization
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transforms.Compose(transform_list)


class MixUp:
    """
    MixUp augmentation for images and labels.
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying mixup
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, num_classes)
            
        Returns:
            mixed_images: Mixed images
            mixed_labels: Mixed labels
        """
        if np.random.random() > self.p:
            return images, labels
        
        batch_size = images.shape[0]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size, device=images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels


class CutMix:
    """
    CutMix augmentation for images and labels.
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying cutmix
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, num_classes)
            
        Returns:
            mixed_images: Mixed images
            mixed_labels: Mixed labels
        """
        if np.random.random() > self.p:
            return images, labels
        
        batch_size = images.shape[0]
        H, W = images.shape[2:]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size, device=images.device)
        
        # Generate random box
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels


class RandomAugmentChestXray:
    """
    Random augmentation specifically designed for chest X-rays.
    """
    
    def __init__(self, n: int = 2, m: int = 10):
        """
        Args:
            n: Number of augmentations to apply
            m: Magnitude of augmentations (0-10)
        """
        self.n = n
        self.m = m
        self.augmentations = self._get_augmentations()
    
    def _get_augmentations(self) -> List[Callable]:
        """Get list of augmentations suitable for chest X-rays."""
        return [
            lambda img: A.RandomBrightnessContrast(
                brightness_limit=self.m/20,
                contrast_limit=self.m/20,
                p=1
            )(image=img)['image'],
            lambda img: A.RandomGamma(
                gamma_limit=(100-self.m*2, 100+self.m*2),
                p=1
            )(image=img)['image'],
            lambda img: A.ShiftScaleRotate(
                shift_limit=self.m/100,
                scale_limit=self.m/100,
                rotate_limit=self.m,
                p=1
            )(image=img)['image'],
            lambda img: A.ElasticTransform(
                alpha=self.m*5,
                sigma=self.m*2,
                p=1
            )(image=img)['image'],
            lambda img: A.GridDistortion(
                distort_limit=self.m/30,
                p=1
            )(image=img)['image'],
        ]
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Randomly select n augmentations
        selected_augs = np.random.choice(
            self.augmentations,
            size=self.n,
            replace=False
        )
        
        # Apply selected augmentations
        for aug in selected_augs:
            image = aug(image)
        
        return image