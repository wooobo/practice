import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from wandb_mnist.wandb_starter.data_loader.default_loader import MNIST, MNISTDataset
from wandb_mnist.wandb_starter.evaluate import evaluation
from wandb_mnist.wandb_starter.net.CNN import CNN
from wandb_mnist.wandb_starter.train import training_epoch_loop
from wandb_mnist.wandb_starter.wandb_utils import load_config, get_device, init_wandb


def load_object(module, name, **kwargs):
    """문자열로 된 클래스 이름을 모듈에서 찾아 객체 생성"""
    return getattr(module, name)(**kwargs)


def main():
    # 설정 불러오기
    config = load_config(config_path="config/default.yml")

    # 데이터셋 설정
    mnist = MNIST()
    train_dataset = MNISTDataset(mnist.dataset['train_img'], mnist.dataset['train_label'])
    test_dataset = MNISTDataset(mnist.dataset['test_img'], mnist.dataset['test_label'])

    # 데이터 로더 설정
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=config["dataset"]["shuffle"]
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False
    )

    # 모델 설정
    device = get_device()
    model = CNN().to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = load_object(nn, config["training"]["criterion"])  # e.g., "CrossEntropyLoss"
    optimizer = load_object(optim, config["training"]["optimizer"], params=model.parameters(),
                            lr=config["training"]["learning_rate"])

    # wandb 초기화
    wandb = init_wandb(config)
    wandb.watch(model)

    # 학습 실행
    training_epoch_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        criterion_name=config["training"]["criterion"],
        optimizer=optimizer,
        device=device,
        evaluation=evaluation,
        epochs=config["training"]["epochs"],
        log_fn=wandb.log
    )


if __name__ == "__main__":
    main()
