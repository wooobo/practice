import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import wandb

from wandb_mnist.wandb_starter.data_loader.default_loader import MNIST, MNISTDataset
from wandb_mnist.wandb_starter.evaluate import evaluation
from wandb_mnist.wandb_starter.net.CNN import CNN
from wandb_mnist.wandb_starter.train import training_epoch_loop
from wandb_mnist.wandb_starter.wandb_utils import load_config, get_device, init_wandb


def load_object(module, name, **kwargs):
    """문자열로 된 클래스 이름을 모듈에서 찾아 객체 생성"""
    return getattr(module, name)(**kwargs)


def train():
    # 🏁 wandb 실행 및 설정 가져오기
    wandb.init()
    config = wandb.config

    # 📌 데이터셋 설정
    mnist = MNIST()
    train_dataset = MNISTDataset(mnist.dataset['train_img'], mnist.dataset['train_label'])
    test_dataset = MNISTDataset(mnist.dataset['test_img'], mnist.dataset['test_label'])

    # 📌 데이터 로더 설정
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # 📌 모델 설정
    device = get_device()
    model = CNN().to(device)

    # 📌 손실 함수 및 옵티마이저 설정
    criterion = load_object(nn, config.loss_function)  # e.g., "CrossEntropyLoss"
    optimizer = load_object(optim, config.activation, params=model.parameters(), lr=config.learning_rate)

    # 📌 wandb 초기화 및 모델 감시
    wandb.watch(model)

    # 📌 학습 실행
    training_epoch_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        criterion_name=config.loss_function,
        optimizer=optimizer,
        device=device,
        evaluation=evaluation,
        epochs=config.epochs,
        log_fn=wandb.log
    )


if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',  # 모든 조합을 시도하는 Grid Search
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'model': {'values': ['ResNet50', 'EfficientNet-B3', 'ConvNeXt-T', 'SwinTransformer-T']},
            'activation': {'values': ['Adam', 'RMSprop', 'SGD']},
            'loss_function': {'values': ['CrossEntropyLoss', 'MSELoss']},
            'augmentation': {'values': ['weak', 'moderate', 'strong']},
            'batch_size': {'values': [16, 32, 64]},  # 배치 크기 조정
            'epochs': {'values': [1, 2, 3]},  # 실험 가능한 에폭 수 조정
            "learning_rate": {"values": [0.01, 0.001, 0.0001]},
        }
    }

    # 🏁 Sweep 실행
    sweep_id = wandb.sweep(sweep_config, project="document_classification")

    # 🏁 여러 조합 실험 (자동으로 10번 실행)
    wandb.agent(sweep_id, function=train, count=100)
