import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

# 커스텀 모듈 import
from rl_dataset import RLPretrainDataset


class NumpyToTensor:
    """NumPy 배열 (H, W, C)를 Tensor (C, H, W)로 변환"""

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # (H, W, C) -> (C, H, W), normalize to [0, 1]
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            if img.max() > 1.0:
                img = img / 255.0
            return img
        return img


class DataToTensor:
    """NumPy 배열 (B, H, W, C)를 Tensor (B, C, H, W)로 변환 (배치 포함)"""

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            # (B, H, W, C) -> (B, C, H, W), normalize to [0, 1]
            data = torch.from_numpy(data).permute(0, 3, 1, 2).float()
            if data.max() > 1.0:
                data = data / 255.0
            return data
        return data


class ColorJitterMultiChannel:
    """12채널 이미지에 대한 ColorJitter (3채널씩 적용)"""

    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1):
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )

    def __call__(self, img):
        """
        Args:
            img: Tensor (12, H, W) - 4개 프레임, 각 3채널
        """
        if img.shape[0] == 12:
            # 3채널씩 분리하여 처리
            frames = []
            for i in range(4):
                frame = img[i * 3 : (i + 1) * 3]  # (3, H, W)
                frame = self.jitter(frame)
                frames.append(frame)
            return torch.cat(frames, dim=0)  # (12, H, W)
        else:
            # 12채널이 아니면 그대로 반환
            return img


def create_data_loaders(data_path, config):
    """데이터 로더 생성"""
    w, h = config["image_size"]
    # 변환 함수
    train_transform = get_default_transforms(
        image_size=(h, w),
        augmentation=True,
    )
    val_transform = get_default_transforms(image_size=(h, w), augmentation=False)

    # 전체 데이터셋 로드 (transform 없이)
    full_dataset = RLPretrainDataset(data_path, transform=None)

    # 학습/검증 분할
    val_split = config.get("val_split", 0.2)
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 각 데이터셋에 맞는 transform 적용
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        persistent_workers=config.get("num_workers", 0) > 0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        persistent_workers=config.get("num_workers", 0) > 0,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_default_transforms(image_size=(320, 180), augmentation=False):
    """
    기본 이미지 변환 함수 생성 (12채널 지원)

    Args:
        image_size (tuple): 이미지 크기 (H, W)
        augmentation (bool): 데이터 증강 사용 여부

    Returns:
        transforms.Compose: 변환 함수
    """
    transform_list = [
        NumpyToTensor(),  # NumPy (H, W, 12) -> Tensor (12, H, W)
    ]

    if augmentation:
        transform_list.append(
            ColorJitterMultiChannel(brightness=0.1, contrast=0.1, saturation=0.1)
        )

    # Resize는 12채널 모두에 적용됨
    transform_list.append(transforms.Resize(image_size))

    return transforms.Compose(transform_list)


def calculate_class_weights(data_path, method="inverse"):
    """
    데이터셋의 클래스별 샘플 개수에 따라 가중치 계산

    Args:
        data_path (str): 데이터셋 경로
        method (str): 가중치 계산 방법
            - "inverse": 1 / count (역수)
            - "inverse_sqrt": 1 / sqrt(count)
            - "effective": effective number 기반 (beta=0.9999)

    Returns:
        torch.Tensor: 클래스별 가중치 텐서
    """
    # 데이터셋 로드하여 클래스 분포 확인
    dataset = RLPretrainDataset(data_path, transform=None)
    class_distribution = dataset.get_class_distribution()

    # 클래스 순서대로 샘플 개수 리스트 생성
    num_classes = len(dataset.classes)
    class_counts = [class_distribution[class_name] for class_name in dataset.classes]

    print(f"\n클래스별 샘플 개수:")
    for class_name, count in zip(dataset.classes, class_counts):
        print(f"  {class_name}: {count}개")

    # 가중치 계산
    if method == "inverse":
        # 역수 방법: weight = 1 / count
        weights = [1.0 / count for count in class_counts]
    elif method == "inverse_sqrt":
        # 제곱근 역수 방법: weight = 1 / sqrt(count)
        weights = [1.0 / (count**0.5) for count in class_counts]
    elif method == "effective":
        # Effective number 방법 (Class-Balanced Loss)
        beta = 0.9999
        effective_nums = [1.0 - beta**count for count in class_counts]
        weights = [(1.0 - beta) / en for en in effective_nums]
    else:
        raise ValueError(f"Unknown method: {method}")

    # 정규화 (합이 num_classes가 되도록)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes

    print(f"\n계산된 클래스 가중치 (method={method}):")
    for class_name, weight in zip(dataset.classes, weights):
        print(f"  {class_name}: {weight:.4f}")

    return weights
