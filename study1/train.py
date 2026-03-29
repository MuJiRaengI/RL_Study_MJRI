"""
study1/train.py - CartPole 학습 코드
=====================================
CartPole-v1 환경에서 PPO 알고리즘으로 학습합니다.
기본은 CPU 학습이지만, DEVICE 변수를 변경하면 GPU로도 학습 가능합니다.

CartPole이란?
  - 막대(pole)가 달린 카트를 좌우로 움직여서 막대를 쓰러지지 않게 균형 잡는 문제
  - 관측값(state): [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
  - 행동(action): 0 = 왼쪽, 1 = 오른쪽
  - 보상(reward): 매 스텝마다 +1 (막대가 쓰러지거나 카트가 범위를 벗어나면 종료)
  - 최고 점수: 500점 (500 스텝 버티면 성공)
"""

import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ─────────────────────────────────────────────
# 설정값 (필요에 따라 수정하세요)
# ─────────────────────────────────────────────

# 학습 디바이스 설정
#   "cpu"  → CPU로 학습 (CartPole처럼 가벼운 환경에선 CPU가 더 빠를 수 있음)
#   "cuda" → GPU로 학습 (NVIDIA GPU 필요)
DEVICE = "cpu"

# 총 학습 스텝 수
TOTAL_TIMESTEPS = 200_000

# 평가 주기: 이 스텝마다 모델 성능을 측정하고 best 모델 갱신 여부 결정
EVAL_FREQ = 5_000

# 평가 시 에피소드 수 (많을수록 정확하지만 느림)
N_EVAL_EPISODES = 10

# 이 스크립트 파일 위치 기준으로 경로 설정 (어디서 실행해도 항상 study1/models/에 저장됨)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
LOG_PATH = os.path.join(BASE_DIR, "logs")


def make_env():
    """CartPole-v1 환경 생성 함수 (DummyVecEnv에 넘길 lambda 형태로 사용)"""
    def _init():
        env = gym.make("CartPole-v1")
        return env
    return _init


def main():
    print("=" * 50)
    print("CartPole PPO 학습 시작")
    print(f"  디바이스: {DEVICE}")
    print(f"  총 학습 스텝: {TOTAL_TIMESTEPS:,}")
    print("=" * 50)

    # GPU 사용 여부 확인
    if DEVICE == "cuda":
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  경고: CUDA를 사용할 수 없습니다. CPU로 대체합니다.")
            # DEVICE = "cpu"  # 자동 CPU 전환 원하면 주석 해제

    # 저장 경로 생성
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # ─────────────────────────────────────────────
    # 학습 환경 생성
    # ─────────────────────────────────────────────
    # DummyVecEnv: SB3은 Vectorized Environment를 사용함
    # (여러 환경을 병렬로 돌릴 수 있는 구조, 여기선 1개만 사용)
    train_env = DummyVecEnv([make_env()])

    # 평가용 환경 (학습 환경과 별도로 사용)
    eval_env = DummyVecEnv([make_env()])

    # ─────────────────────────────────────────────
    # PPO 모델 생성
    # ─────────────────────────────────────────────
    # MlpPolicy: 다층 퍼셉트론(MLP) 정책 네트워크 (이미지가 아닌 벡터 입력에 사용)
    # verbose=1: 학습 진행 상황 출력
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        device=DEVICE,
        verbose=1,
        # 아래는 PPO의 주요 하이퍼파라미터 (기본값 사용, 튜닝 원하면 수정)
        # n_steps=2048,        # 업데이트 전 수집할 스텝 수
        # batch_size=64,       # 미니배치 크기
        # n_epochs=10,         # 데이터 재사용 횟수
        # learning_rate=3e-4,  # 학습률
        # gamma=0.99,          # 할인율 (미래 보상의 중요도)
    )

    # ─────────────────────────────────────────────
    # EvalCallback 설정
    # ─────────────────────────────────────────────
    # EvalCallback: 주기적으로 평가하고, 성능이 향상되면 best_model 저장
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,  # best_model.zip 저장 위치
        log_path=LOG_PATH,                      # 평가 결과 로그 저장 위치
        eval_freq=EVAL_FREQ,                    # 몇 스텝마다 평가할지
        n_eval_episodes=N_EVAL_EPISODES,        # 평가 에피소드 수
        deterministic=True,                     # 평가 시 확률적이 아닌 결정적 행동 사용
        render=False,
        verbose=1,
    )

    # ─────────────────────────────────────────────
    # 학습 실행
    # ─────────────────────────────────────────────
    print("\n학습을 시작합니다...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    print("\n학습 완료!")
    print(f"  Best 모델 저장 위치: {MODEL_SAVE_PATH}/best_model.zip")

    # 최종 모델도 별도 저장 (best 모델과 다를 수 있음)
    model.save(f"{MODEL_SAVE_PATH}/final_model")
    print(f"  최종 모델 저장 위치: {MODEL_SAVE_PATH}/final_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
