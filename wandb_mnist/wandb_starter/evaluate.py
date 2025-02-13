# # 평가 함수
# import torch
#
#
# def evaluation(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     return accuracy
import torch
from sklearn.metrics import f1_score

def evaluation(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # F1-score를 위해 리스트에 저장
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 정확도 계산
    accuracy = 100 * correct / total

    # F1-score 계산 (다중 클래스일 경우 average='macro' 또는 'weighted' 사용)
    f1 = f1_score(y_true, y_pred, average='macro')  # 'macro'는 클래스별 평균

    return accuracy, f1
