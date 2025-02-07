import numpy as np
import matplotlib.pyplot as plt

def plot_random_samples(X_train, y_train, num_samples=16, img_size=(28, 28), cmap='gray'):
    """
    MNIST 데이터셋에서 랜덤하게 num_samples 개의 샘플을 시각화합니다.

    Parameters:
    - X_train (numpy.ndarray): 학습 데이터 이미지
    - y_train (numpy.ndarray): 학습 데이터 레이블
    - num_samples (int): 표시할 샘플 개수 (기본값: 16)
    - img_size (tuple): 이미지 크기 (기본값: (28, 28))
    - cmap (str): 컬러맵 설정 (기본값: 'gray')

    Returns:
    - None (이미지 출력)
    """
    plt.figure(figsize=(7, 7))
    random_indices = np.random.randint(0, len(X_train), size=num_samples)

    for n, i in enumerate(random_indices, start=1):
        plt.subplot(4, 4, n)
        plt.imshow(X_train[i].reshape(img_size), cmap=cmap)
        plt.title(f"Label: {y_train[i]}", fontsize=12)
        plt.axis('off')

    plt.suptitle('MNIST Dataset', fontsize=20)
    plt.tight_layout()
    plt.show()

