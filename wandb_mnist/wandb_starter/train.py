from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                    log_fn=None,
                    ):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Training")
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    metrics = {"train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1}

    if log_fn:
        log_fn(metrics)

    return metrics


def training_epoch_loop(model,
                        train_loader,
                        test_loader,
                        criterion,
                        optimizer,
                        device,
                        evaluation,
                        epochs,
                        log_fn=None,
                        ):
    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, log_fn)
        test_accuracy = evaluation(model, test_loader, device)

        log_data = {
            "epoch": epoch + 1,
            "test_accuracy": test_accuracy,
            **train_metrics
        }

        if log_fn:
            log_fn(log_data)

        print(f"\n Epoch {epoch + 1}: Loss = {train_metrics['train_loss']:.4f}, "
              f"Train Acc = {train_metrics['train_acc']:.2f}%, "
              f"Train F1 = {train_metrics['train_f1']:.2f}, "
              f"Test Accuracy = {test_accuracy:.2f}%")
