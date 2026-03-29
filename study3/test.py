"""
study3/test.py - Walk the Stork 테스트 코드
============================================
학습이 완료된 모델을 불러와 Walk the Stork 게임을 평가합니다.

study1/2와의 차이점:
  - study1/2는 train.py와 동시에 실행하여 실시간으로 모델 업데이트를 확인합니다.
  - study3는 실제 게임 환경이 1개뿐이므로 train과 test를 동시에 실행할 수 없습니다.
  - 반드시 train.py 학습 완료 후 실행하세요.

사용법:
  1. train.py를 실행하여 학습을 완료합니다.
  2. 학습 완료 후 이 스크립트를 실행합니다.
     python study3/test.py
  3. 게임이 N_EVAL_EPISODES번 실행되고 통계가 출력됩니다.
"""

import os
import sys
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# study3 디렉토리를 sys.path에 추가 (어디서 실행해도 import가 되도록)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from custom_env import WalkTheStorkEnv

# ─────────────────────────────────────────────
# 설정값 (필요에 따라 수정하세요)
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 테스트할 모델 경로
#   None       → 기본값: study3/models/best_model.zip 사용
#   "경로.zip" → 해당 모델로 테스트
TEST_MODEL_PATH = None
# TEST_MODEL_PATH = r"study3\models\final_model.zip"       # 최종 모델로 테스트
TEST_MODEL_PATH = r"MJRI_ppo_bestmodel.zip" # 특정 체크포인트로 테스트

# TEST_MODEL_PATH가 None이면 기본 경로(best_model.zip) 사용
MODEL_PATH = TEST_MODEL_PATH if TEST_MODEL_PATH else os.path.join(BASE_DIR, "models", "best_model.zip")

# 평가할 에피소드 수
N_EVAL_EPISODES = 20

# 환경 설정 (train.py와 동일하게 맞춰야 함)
ENV_FPS      = 30
ENV_CROP_POS = (300, 600, 700, 450)
ENV_RESIZE   = (320, 180)
ENV_STACKED  = 4


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
    )
    env.run_capture()  # 화면 캡처 스레드 시작
    return env


# ─────────────────────────────────────────────
# 에피소드 실행 함수
# ─────────────────────────────────────────────

def run_episode(vec_env, model, obs) -> tuple[float, int, float, np.ndarray]:
    """
    에피소드를 한 번 실행합니다.

    VecEnv를 사용하므로 obs, rewards, dones는 배열 형태입니다.
    에피소드 종료 시 DummyVecEnv가 자동으로 reset을 호출하므로,
    새로운 obs를 반환하여 다음 에피소드에서 사용하도록 합니다.

    Returns:
        (total_reward, steps, distance, next_obs): 에피소드 총 보상, 스텝 수, 진행 거리(m), 다음 에피소드를 위한 obs
    """
    total_reward = 0.0
    steps = 0
    distance = 0.0

    while True:
        # 모델로 행동 선택 (deterministic=True: 탐험 없이 최선의 행동만 선택)
        action, _ = model.predict(obs, deterministic=True)

        # 환경에 행동 적용 (VecEnv는 배열 반환)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += rewards[0]  # VecEnv는 배열이므로 [0]으로 접근
        steps += 1

        # 에피소드 종료 (사망)
        if dones[0]:
            # info에서 진행 거리 추출
            if len(infos) > 0 and "distance" in infos[0]:
                distance = infos[0]["distance"]
            break

    return total_reward, steps, distance, obs


# ─────────────────────────────────────────────
# 메인 평가 함수
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Walk the Stork 테스트 시작")
    print(f"  모델 경로: {MODEL_PATH}")
    print(f"  평가 에피소드 수: {N_EVAL_EPISODES}")
    print("=" * 60)

    # ─────────────────────────────────────────────
    # 모델 로드
    # ─────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\n[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("  먼저 train.py를 실행하여 학습을 완료하세요.")
        return

    print(f"\n모델 로드 중: {MODEL_PATH}")
    # EpisodeBoundaryPPO는 PPO의 서브클래스이므로 PPO.load()로 로드 가능
    # (테스트 시에는 learn()을 사용하지 않으므로 표준 PPO로 로드해도 동일하게 동작)
    model = PPO.load(MODEL_PATH)
    print("모델 로드 완료!")

    # ─────────────────────────────────────────────
    # 환경 생성
    # ─────────────────────────────────────────────
    # 테스트 시에는 VecNormalize 불필요:
    #   - norm_obs=False이므로 관측값이 달라지지 않음
    #   - norm_reward는 학습에만 영향을 주고 행동 선택에는 영향 없음
    vec_env = DummyVecEnv([make_env])

    # ─────────────────────────────────────────────
    # 평가 실행
    # ─────────────────────────────────────────────
    episode_rewards = []
    episode_lengths = []
    episode_distances = []

    print(f"\n{N_EVAL_EPISODES}번 에피소드 평가를 시작합니다...\n")

    try:
        # 최초 1회만 리셋 (이후에는 DummyVecEnv가 done 시 자동 리셋함)
        obs = vec_env.reset()

        for episode in range(1, N_EVAL_EPISODES + 1):
            total_reward, steps, distance, obs = run_episode(vec_env, model, obs)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_distances.append(distance)

            print(
                f"  에피소드 {episode:3d}/{N_EVAL_EPISODES} | "
                f"보상: {total_reward:8.2f} | "
                f"스텝: {steps:5d} | "
                f"거리: {distance:6.2f}m"
            )

    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")

    # ─────────────────────────────────────────────
    # 통계 출력
    # ─────────────────────────────────────────────
    if episode_rewards:
        print("\n" + "=" * 60)
        print("평가 결과 통계")
        print("=" * 60)
        print(f"  평균 보상:    {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  평균 스텝:    {np.mean(episode_lengths):8.1f} ± {np.std(episode_lengths):.1f}")
        if any(d > 0 for d in episode_distances):
            print(f"  평균 거리:    {np.mean(episode_distances):8.2f}m ± {np.std(episode_distances):.2f}m")
            print(f"  최대 거리:    {np.max(episode_distances):8.2f}m")
        print(f"  최고 보상:    {np.max(episode_rewards):8.2f}")
        print(f"  최저 보상:    {np.min(episode_rewards):8.2f}")
        print(f"  평가 에피소드: {len(episode_rewards)}개")
        print("=" * 60)

    vec_env.close()


if __name__ == "__main__":
    main()
