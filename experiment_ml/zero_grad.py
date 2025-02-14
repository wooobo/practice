import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 모델 정의
model = nn.Linear(2, 1)  # 입력 2차원, 출력 1차원
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# 임의의 입력 데이터
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
y_true = torch.tensor([[5.0]])

# zero_grad() 없이 실행
optimizer.zero_grad()  # 그래디언트 초기화
y_pred = model(x)
loss = criterion(y_pred, y_true)
loss.backward()
print("첫 번째 backward 후 가중치의 기울기:", model.weight.grad)

y_pred = model(x)  # 다시 forward
loss = criterion(y_pred, y_true)
loss.backward()  # 그래디언트가 누적됨 (zero_grad() 안 했음)
print("두 번째 backward 후 가중치의 기울기:", model.weight.grad)

print("zero_grad 적용")
# zero_grad() 없이 실행
optimizer.zero_grad()  # 그래디언트 초기화
y_pred = model(x)
loss = criterion(y_pred, y_true)
loss.backward()
print("첫 번째 backward 후 가중치의 기울기:", model.weight.grad)

optimizer.zero_grad()  # 그래디언트 초기화
y_pred = model(x)  # 다시 forward
loss = criterion(y_pred, y_true)
loss.backward()  # 그래디언트가 누적됨 (zero_grad() 안 했음)
print("두 번째 backward 후 가중치의 기울기:", model.weight.grad)
