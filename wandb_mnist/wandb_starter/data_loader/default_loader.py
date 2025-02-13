import numpy as np
import os
import gzip

import torch
import torch.utils.data as data

class MNIST:
    def __init__(self):
        self.model = self
        self.file_paths = {
            'train_img': 'train-images-idx3-ubyte.gz',
            'train_label': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz',
            'test_label': 't10k-labels-idx1-ubyte.gz'
        }
        self.data_dir = 'data'  # 데이터 폴더 경로
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """데이터셋을 로드하는 함수"""
        dataset = {}
        for key, value in self.file_paths.items():
            if 'img' in key:
                dataset[key] = self.load_images(value)
            else:
                dataset[key] = self.load_labels(value)
        return dataset

    def load_labels(self, filename):
        """레이블 파일을 로드하는 함수"""
        path = os.path.join(self.data_dir, filename)
        with gzip.open(path, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return labels

    def load_images(self, filename):
        """이미지 파일을 로드하는 함수"""
        path = os.path.join(self.data_dir, filename)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            images = data.reshape(-1, 28, 28)  # (N, 28, 28) 형식으로 변환
        return images

# 데이터 로더 클래스
class MNISTDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # 정규화
        img = np.expand_dims(img, axis=0)  # (28, 28) -> (1, 28, 28)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
