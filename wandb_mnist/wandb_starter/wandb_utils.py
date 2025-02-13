import datetime

import wandb
import torch
import yaml

# Wandb Initialization
def init_wandb(config, project_name="image_classification_base"):
    """
    WandB 프로젝트 초기화
    :param config: config 파일
    :return: wandb 객체
    """
    if config["wandb"]["use_wandb"]:
        print('group name ==> ' , config["experiment"]["group_name"])
        wandb.init(
            project=config["experiment"]["project_name"],
            group=config["experiment"]["group_name"],
            name=generate_experiment_name(config),
            config=config
        )
        print(f"[WandB] 프로젝트 '{project_name}' 로깅 시작")

    return wandb

# Wandb Metric Logging
def log_metrics(metrics, step=None):
    """학습 중 주요 메트릭 로깅"""
    if wandb.run is not None:
        wandb.log(metrics, step=step)

# Wandb Artifact를 활용한 모델 저장
def save_model(model, model_path="models/mnist_model.pth", artifact_name="mnist_model"):
    """WandB Artifact를 활용한 모델 저장"""
    torch.save(model.state_dict(), model_path)
    if wandb.run is not None:
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        print(f"[WandB] 모델 '{artifact_name}' 저장 완료")

# Wandb Commit
def finish_wandb():
    """WandB 세션 종료"""
    if wandb.run is not None:
        wandb.finish()
        print("[WandB] 로깅 종료 완료")

# 실험 환경 이름 생성
def generate_experiment_name(config):
    return f"{config['experiment']['experiment_name']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# YAML 설정 불러오기
def load_config(config_path="config/default.yml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# ---------------------------------------------------------------------------------
# ---------------------------------- Environment ----------------------------------
# ---------------------------------------------------------------------------------
def get_device(config=None):
    return torch.device("cuda" if torch.cuda.is_available() and config["device"]["use_cuda"] else "cpu")