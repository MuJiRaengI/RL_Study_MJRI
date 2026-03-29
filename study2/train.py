"""
study2/train.py - Breakout 학습 코드 (GPU)
============================================
Atari Breakout 게임을 CNN + PPO로 학습합니다.

Breakout이란?
  - 공을 튕겨서 위쪽 벽돌을 모두 부수는 게임
  - 관측값(state): 게임 화면 이미지 (84x84 픽셀, 4프레임 스택)
  - 행동(action): 0=정지, 1=발사, 2=오른쪽, 3=왼쪽
  - 보상(reward): 벽돌을 부술 때마다 +1

사전 준비:
  pip install stable-baselines3[extra]
  pip install autorom[accept-rom-license]  # Atari ROM 라이선스 동의 후 설치
  AutoROM --accept-license                 # ROM 다운로드

Atari 전처리 (SB3 내장):
  - 프레임 스킵: 4프레임마다 행동 1회
  - 그레이스케일 변환
  - 84x84 리사이즈
  - 프레임 4개 스택 (동적 정보 포착용)
"""

import os
import torch
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO

# Atari 환경을 gymnasium에 등록 (없으면 BreakoutNoFrameskip 등을 찾지 못함)
gym.register_envs(ale_py)
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────

# 학습 디바이스 (Atari는 이미지 처리라 GPU가 훨씬 빠름)
DEVICE = "cuda"  # CPU로 학습하려면 "cpu"로 변경

# 환경 이름 (NoFrameskip 버전 사용, SB3이 직접 프레임 스킵 처리)
ENV_ID = "BreakoutNoFrameskip-v4"

# 병렬 환경 수 (많을수록 빠르지만 메모리 많이 사용)
N_ENVS = 8

# 프레임 스택 수 (연속 4프레임으로 속도/방향 정보 포착)
N_STACK = 4

# 총 학습 스텝 수 (Atari는 10M 이상 학습해야 의미 있는 성능이 나옴)
TOTAL_TIMESTEPS = 10_000_000

# 평가 주기
# EvalCallback의 eval_freq는 VecEnv step 횟수(n_calls) 기준임
# N_ENVS=8이면 VecEnv 1 step = 실제 8 timesteps이므로,
# "50,000 timesteps마다 평가"하려면 50_000 // N_ENVS 로 설정해야 함
EVAL_FREQ = 50_000 // N_ENVS  # 실제 50,000 timesteps마다 평가

# 평가 에피소드 수
N_EVAL_EPISODES = 5

# 이 스크립트 파일 위치 기준으로 경로 설정 (어디서 실행해도 항상 study2/models/에 저장됨)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
LOG_PATH = os.path.join(BASE_DIR, "logs")


def main():
    print("=" * 50)
    print("Breakout PPO 학습 시작")
    print(f"  디바이스: {DEVICE}")
    print(f"  병렬 환경 수: {N_ENVS}")
    print(f"  총 학습 스텝: {TOTAL_TIMESTEPS:,}")
    print("=" * 50)

    # GPU 확인
    if DEVICE == "cuda":
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  경고: CUDA를 사용할 수 없습니다. CPU로 대체합니다.")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # ─────────────────────────────────────────────
    # Atari 환경 생성
    # ─────────────────────────────────────────────
    # make_atari_env: Atari에 필요한 전처리 래퍼를 자동으로 적용
    #   - NoopResetEnv: 시작 시 랜덤 대기 (과적합 방지)
    #   - MaxAndSkipEnv: 4프레임 스킵 + 최댓값 프레임 사용
    #   - EpisodicLifeEnv: 목숨 잃을 때마다 에피소드 종료 처리
    #   - WarpFrame: 84x84 그레이스케일 변환
    #   - ClipRewardEnv: 보상을 -1, 0, +1로 클리핑
    train_env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=42)
    # VecFrameStack: N개 연속 프레임을 쌓아서 속도/방향 정보를 포착
    train_env = VecFrameStack(train_env, n_stack=N_STACK)

    # 평가용 환경 (1개, 병렬 없이 순수 평가용)
    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

    # ─────────────────────────────────────────────
    # PPO 모델 생성
    # ─────────────────────────────────────────────
    # CnnPolicy: 이미지 입력을 위한 CNN 정책 네트워크
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        device=DEVICE,
        verbose=1,
        # Atari에 맞게 조정된 하이퍼파라미터
        n_steps=128,         # 짧은 롤아웃 (병렬 환경이 많으므로)
        batch_size=256,      # 배치 크기
        n_epochs=4,          # 에폭 수
        learning_rate=2.5e-4,
        clip_range=0.1,      # PPO 클리핑 범위
        vf_coef=0.5,         # 가치 함수 손실 가중치
        ent_coef=0.01,       # 엔트로피 보너스 (탐색 장려)
        gamma=0.99,          # 할인율
        gae_lambda=0.95,     # GAE 람다
    )

    # ─────────────────────────────────────────────
    # EvalCallback 설정
    # ─────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,
        log_path=LOG_PATH,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ─────────────────────────────────────────────
    # 학습 실행
    # ─────────────────────────────────────────────
    print("\n학습을 시작합니다...")
    print("  (Atari 학습은 시간이 많이 걸립니다. GPU를 권장합니다)")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
        log_interval=50,  # 50 iteration마다 한 번 출력 (기본값 1 → 매우 자주 출력됨)
    )

    print("\n학습 완료!")
    print(f"  Best 모델 저장 위치: {MODEL_SAVE_PATH}/best_model.zip")

    model.save(f"{MODEL_SAVE_PATH}/final_model")
    print(f"  최종 모델 저장 위치: {MODEL_SAVE_PATH}/final_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
