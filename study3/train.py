"""
study3/train.py - Walk the Stork 학습 코드 (GPU)
=================================================
Walk the Stork 실제 게임 환경에서 PPO 알고리즘으로 학습합니다.

study1/2와의 차이점:
  1. 에피소드 경계 학습 (ppo.py - EpisodeBoundaryPPO):
     - 표준 PPO는 n_steps마다 즉시 학습을 시작하지만,
       실제 게임 중에 학습을 시작하면 에피소드가 중단될 수 있습니다.
     - EpisodeBoundaryPPO는 충분한 데이터(MIN_TRAIN_STEPS)가 모여도
       현재 에피소드(게임 한 판)가 끝날 때까지 기다린 후 학습을 시작합니다.

  2. train/test 순차 실행:
     - 실제 게임은 환경이 1개뿐이므로 train과 test를 동시에 실행할 수 없습니다.
     - train.py로 학습을 완료한 후 test.py를 실행하세요.
     - BestModelCallback(callbacks.py)이 최고 성능 모델을 best_model.zip으로 저장합니다.

파일 구조:
  custom_env.py  → WalkTheStorkEnv (게임 환경 정의)
  capture.py     → GameCapture (화면 캡처 스레드)
  consts.py      → 액션/상태 상수
  ppo.py         → EpisodeBoundaryPPO (에피소드 경계 학습)
  callbacks.py   → BestModelCallback (최고 모델 저장)
  train.py       → 학습 설정 및 실행 (이 파일)
  test.py        → 평가 실행
"""

import os
import sys
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# study3 디렉토리를 sys.path에 추가 (어디서 실행해도 import가 되도록)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from custom_env import WalkTheStorkEnv
from ppo import EpisodeBoundaryPPO
from callbacks import BestModelCallback

# ─────────────────────────────────────────────
# 설정값 (필요에 따라 수정하세요)
# ─────────────────────────────────────────────

# 학습 디바이스
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 총 학습 스텝 수
TOTAL_TIMESTEPS = 10_000_000

# 에피소드 경계 학습: 이 스텝 이상 데이터가 모이면, 현재 에피소드 종료 후 학습 시작
# (표준 PPO의 n_steps와 다르게, 에피소드 중간에 학습이 끊기지 않습니다)
MIN_TRAIN_STEPS = 2048

# PPO의 rollout buffer 크기 (MIN_TRAIN_STEPS보다 충분히 크게 설정)
# 하나의 에피소드가 아무리 길어도 이 크기를 넘기면 버퍼가 가득 참
N_STEPS = 16384

# 체크포인트 저장 주기 (이 스텝마다 중간 모델 저장)
CHECKPOINT_FREQ = 50_000

# best_model 갱신 판단 기준: 최근 몇 에피소드의 평균 보상을 사용할지
BEST_MODEL_WINDOW = 5

# 사전학습 모델 경로
#   None       → 처음부터 학습 (랜덤 초기화)
#   "경로.zip" → 해당 모델의 weight를 불러와서 이어서 학습
#   파일이 존재하지 않으면 처음부터 학습합니다
PRETRAINED_MODEL_PATH = None
PRETRAINED_MODEL_PATH = r"MJRI_ppo_bestmodel.zip"  # 예시

# 저장 경로 (이 스크립트 파일 위치 기준)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
LOG_PATH = os.path.join(BASE_DIR, "logs")

# ─────────────────────────────────────────────
# 환경 설정값
# ─────────────────────────────────────────────
ENV_FPS        = 30
ENV_CROP_POS   = (300, 600, 700, 450)  # (x, y, width, height) - 게임 화면 크롭 영역
ENV_RESIZE     = (320, 180)            # 크롭 후 리사이즈 크기 (width, height)
ENV_STACKED    = 4                     # 스택할 프레임 수


# ─────────────────────────────────────────────
# 환경 생성 함수
# ─────────────────────────────────────────────

def make_env():
    """WalkTheStorkEnv 환경 생성 및 캡처 시작."""
    env = WalkTheStorkEnv(
        fps=ENV_FPS,
        crop_pos=ENV_CROP_POS,
        resize=ENV_RESIZE,
        action_num=3,
        stacked_num=ENV_STACKED,
        gray_scale=True,
        device=DEVICE,
    )
    env.run_capture()  # 화면 캡처 스레드 시작 (반드시 호출)
    return env


# ─────────────────────────────────────────────
# 메인 학습 함수
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Walk the Stork PPO 학습 시작")
    print(f"  디바이스: {DEVICE}")
    print(f"  총 학습 스텝: {TOTAL_TIMESTEPS:,}")
    print(f"  에피소드 경계 학습 (min_train_steps: {MIN_TRAIN_STEPS:,})")
    print("=" * 60)

    # GPU 정보 출력
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  CPU 학습 (CUDA를 사용할 수 없음)")

    # 저장 경로 생성
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # ─────────────────────────────────────────────
    # 환경 생성
    # ─────────────────────────────────────────────
    # DummyVecEnv: SB3는 VecEnv 형식을 필요로 함 (단일 환경이므로 1개만)
    train_env = DummyVecEnv([make_env])

    # VecNormalize: 보상 범위가 크기 때문에 reward normalization 적용
    #   생존 시: +0.5 ~ -1.5  / 사망 시: -500 + 진행거리
    #   norm_obs=False: 이미지 관측값은 CNN이 직접 처리 (정규화 불필요)
    #   norm_reward=True: 큰 범위의 보상을 표준화하여 학습 안정화
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    # ─────────────────────────────────────────────
    # 모델 생성
    # ─────────────────────────────────────────────
    # CnnPolicy: 이미지 관측값을 처리하는 CNN 기반 정책 네트워크
    # (study1/2의 MlpPolicy와 달리 합성곱 레이어로 이미지를 처리)
    model = EpisodeBoundaryPPO(
        policy="CnnPolicy",
        env=train_env,
        device=DEVICE,
        verbose=1,
        tensorboard_log=LOG_PATH,
        n_steps=N_STEPS,          # rollout buffer 크기 (수집 중 최대 저장 가능 스텝)
        learning_rate=3e-6,       # 낮은 학습률: 실시간 게임은 안정적 학습 필요
        batch_size=64,            # 미니배치 크기
        n_epochs=8,               # 에폭 수: 같은 데이터를 8번 재사용
        gamma=0.995,              # 높은 할인율: 장기 보상을 중시
        gae_lambda=0.95,          # GAE 람다: 편향-분산 트레이드오프 조절
        clip_range=0.2,           # PPO 클리핑 범위: 너무 큰 업데이트 방지
        vf_coef=0.5,              # 가치 함수 손실 가중치
        ent_coef=0.01,            # 엔트로피 보너스: 충분한 탐험 유도
        max_grad_norm=0.5,        # 그래디언트 클리핑: 학습 안정화
    )

    # ─────────────────────────────────────────────
    # 사전학습 모델 weight 로드 (선택 사항)
    # ─────────────────────────────────────────────
    # PRETRAINED_MODEL_PATH가 설정되어 있고 파일이 존재하면
    # 해당 모델의 policy network weight를 현재 모델에 복사합니다.
    #
    # ※ 전체 모델 복원이 아닌 weight 복사 방식을 사용하는 이유:
    #    - 학습률, n_steps 등 하이퍼파라미터는 위에서 설정한 값을 유지
    #    - 신경망 가중치(CNN 특성 추출기, 정책 헤드)만 가져옴
    #    - strict=False: 레이어 구조가 조금 달라도 호환 가능한 부분만 로드
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"\n사전학습 모델 로드 중: {PRETRAINED_MODEL_PATH}")
        pretrained = PPO.load(PRETRAINED_MODEL_PATH, env=None, device=DEVICE)
        model.policy.load_state_dict(pretrained.policy.state_dict(), strict=False)
        print("사전학습 모델 weight 로드 완료!\n")
        del pretrained  # 메모리 해제
    elif PRETRAINED_MODEL_PATH:
        print(f"\n[경고] 사전학습 모델 파일을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        print("처음부터 학습을 시작합니다.\n")

    # ─────────────────────────────────────────────
    # 콜백 설정
    # ─────────────────────────────────────────────
    # 1. 체크포인트 콜백: 주기적으로 중간 모델 저장 (복구용)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_SAVE_PATH,
        name_prefix="model",
        verbose=1,
    )

    # 2. 최고 모델 콜백: 평균 보상이 최고치 갱신 시 best_model.zip 저장
    best_model_callback = BestModelCallback(
        save_path=MODEL_SAVE_PATH,
        window=BEST_MODEL_WINDOW,
        verbose=1,
    )

    callback_list = CallbackList([checkpoint_callback, best_model_callback])

    # ─────────────────────────────────────────────
    # 학습 실행
    # ─────────────────────────────────────────────
    print("\n학습을 시작합니다. Ctrl+C로 중단할 수 있습니다.\n")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            min_train_steps=MIN_TRAIN_STEPS,
            callback=callback_list,
        )
    except KeyboardInterrupt:
        print("\n\n학습이 중단되었습니다. 현재 모델을 저장합니다...")

    # ─────────────────────────────────────────────
    # 최종 모델 저장
    # ─────────────────────────────────────────────
    model.save(os.path.join(MODEL_SAVE_PATH, "final_model"))
    print(f"\n학습 완료!")
    print(f"  Best 모델: {MODEL_SAVE_PATH}/best_model.zip")
    print(f"  최종 모델: {MODEL_SAVE_PATH}/final_model.zip")
    print(f"\n이제 test.py를 실행하여 학습된 모델을 평가하세요.")

    train_env.close()


if __name__ == "__main__":
    main()
