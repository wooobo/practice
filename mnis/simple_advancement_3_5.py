import gzip
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from mnis.utils import plot_random_samples

# MNIST를 저장할 디렉토리 (`./data/`)
dataset_dir = os.path.join(os.getcwd(), 'data')

# Pickle로 저장할 경로
save_file = dataset_dir + "/mnist.pkl"

# MNIST 데이터셋의 파일명 (딕셔너리)
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

def _load_label(file_name):
    """MNIST 데이터셋 라벨을 NumPy Array로 변환하여 불러오기
    """
    file_path = dataset_dir + "/" + file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def _load_img(file_name):
    """MNIST 데이터셋 이미지을 NumPy Array로 변환하여 불러오기
    """
    file_path = dataset_dir + "/" + file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28*28)
    return data

def _convert_numpy():
    """NumPy Array로 불러온 MNIST 데이터셋을 딕셔너리로 매핑
    """
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    return dataset

def init_mnist():
    """MNIST 데이터셋을 Pickle화
    """
    dataset = _convert_numpy()
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    """
    # Pickle화 됐는지 확인
    if not os.path.exists(save_file):
        init_mnist()

    # Pickle화된 MNIST 데이터셋 가져오기
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 레이블을 원-핫(one-hot) 배열로 변환
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    # 입력 이미지를 1차원 배열로 만듬
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

import numpy as np

def shuffle_and_split_data(X_train, y_train, valid_size=10000):
    """
    학습 데이터(X_train, y_train)를 랜덤하게 섞고, 일부를 검증 데이터(valid)로 분할합니다.

    Parameters:
    - X_train (numpy.ndarray): 학습 데이터 입력
    - y_train (numpy.ndarray): 학습 데이터 레이블
    - valid_size (int): 검증 데이터 개수 (기본값: 10000)

    Returns:
    - X_train (numpy.ndarray): 분할된 학습 데이터
    - y_train (numpy.ndarray): 분할된 학습 레이블
    - X_valid (numpy.ndarray): 검증 데이터
    - y_valid (numpy.ndarray): 검증 레이블
    """
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)  # 인덱스 랜덤 셔플

    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:]

    X_valid, y_valid = X_train[valid_idx], y_train[valid_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    return X_train, y_train, X_valid, y_valid


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_train, y_train, X_valid, y_valid = shuffle_and_split_data(X_train, y_train)

    # MNIST 데이터셋 살펴보기
    print(f"학습 데이터: {len(X_train):,}개\n검증 데이터: {len(X_valid):,}개\n평가 데이터: {len(X_test):,}개")

    plot_random_samples(X_train, y_train)
