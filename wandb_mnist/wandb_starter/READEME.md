### 📌 MNIST 분류 프로젝트 (CNN 기반)  

**Weights & Biases (WandB)**를 활용하여 실험을 추척 프로젝트 구조 예시입니다.  

---

## 📁 프로젝트 구조  

```
wandb_mnist/
├── project/
│   ├── CLI.py                  # 메인 실행 파일
│   ├── config/
│   │   └── default.yml          # 설정 파일
│   ├── wandb_starter/
│   │   ├── data_loader/
│   │   │   └── default_loader.py # MNIST 데이터 로더
│   │   ├── evaluate.py          # 모델 평가 코드
│   │   ├── net/
│   │   │   └── CNN.py           # CNN 모델 정의
│   │   ├── train.py             # 모델 학습 함수
│   │   └── wandb_utils.py       # WandB 설정 및 로깅 함수
```

---

## 🛠️ 필수 환경  

🔹 Python 3.x  
🔹 PyTorch  
🔹 WandB  
🔹 scikit-learn  
🔹 tqdm  
🔹 pyyaml  

---

## 📥 설치 방법  

1️⃣ **프로젝트 클론 및 이동**  
```sh
git clone https://github.com/yourusername/wandb_mnist.git
cd wandb_mnist
```

2️⃣ **필요한 패키지 설치**  
```sh
pip install -r requirements.txt
```

---

## ⚙️ 설정 파일  

설정 파일은 `wandb_mnist/project/config/default.yml`에 위치합니다.  
➡️ **학습 파라미터(에포크, 배치 크기, 학습률 등) 및 WandB 설정**을 수정할 수 있습니다.  

---

## 🚀 실행 방법  

학습을 실행하려면 다음 명령어를 입력하세요.  
```sh
python wandb_mnist/project/CLI.py
```

### ✅ 실행 과정  
1️⃣ 설정 파일(`config/default.yml`) 로드  
2️⃣ MNIST 데이터셋 로드 및 전처리  
3️⃣ CNN 모델 및 손실 함수, 옵티마이저 설정  
4️⃣ **WandB를 이용한 실험 추적**  
5️⃣ 모델 학습 및 메트릭 로깅  

---

## 📂 주요 파일 설명  

| 파일명 | 설명 |
|--------|------|
| `CLI.py` | 학습 실행을 위한 메인 스크립트 |
| `config/default.yml` | 실험 설정 파일 |
| `data_loader/default_loader.py` | MNIST 데이터셋 로드 및 전처리 |
| `evaluate.py` | 학습된 모델을 테스트 데이터셋에서 평가 |
| `net/CNN.py` | CNN 모델 구조 정의 |
| `train.py` | 모델 학습을 위한 함수 포함 |
| `wandb_utils.py` | WandB 설정 및 로깅 유틸리티 |

---
