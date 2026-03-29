"""
study1/test.py - CartPole 테스트 코드
======================================
train.py와 동시에 실행하면 학습 과정을 실시간으로 확인할 수 있습니다.

핵심 기능:
  - best_model.zip이 없으면 랜덤 초기화 모델로 시작 (학습 전 상태)
  - best_model.zip이 생기거나 업데이트되면 자동으로 재로드
  - 에피소드 결과(점수, 스텝 수) 출력
  - 렌더링 ON/OFF 선택 가능

사용법:
  1. train.py와 test.py를 동시에 실행하세요
  2. 처음에는 랜덤 행동으로 금방 쓰러지다가, 학습이 진행될수록 점점 잘 버팁니다
"""

import os
import time
import gymnasium as gym
from stable_baselines3 import PPO

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────

# 이 스크립트 파일 위치 기준으로 경로 설정 (어디서 실행해도 항상 study1/models/에서 로드)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "best_model.zip")

# 렌더링 여부 (True면 화면에 CartPole 시각화, False면 빠른 테스트)
RENDER = True

# 모델 파일 변경 감지 주기 (초)
MODEL_CHECK_INTERVAL = 2.0

# 에피소드 간 대기 시간 (초, 렌더링 시 너무 빠르게 넘어가는 것 방지)
EPISODE_DELAY = 0.5


def load_model(model_path: str) -> PPO:
    """모델을 로드하고 파일의 수정 시간을 함께 반환합니다."""
    print(f"  모델 로드 중: {model_path}")
    model = PPO.load(model_path)
    mtime = os.path.getmtime(model_path)
    print(f"  모델 로드 완료 (수정 시간: {time.ctime(mtime)})")
    return model, mtime


def run_episode(env: gym.Env, model: PPO, render: bool = True) -> tuple[float, int]:
    """
    에피소드를 한 번 실행하고 총 보상과 스텝 수를 반환합니다.

    Returns:
        (total_reward, steps): 에피소드 총 보상, 총 스텝 수
    """
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        # 모델로 행동 선택 (deterministic=True: 가장 높은 확률의 행동 선택)
        action, _ = model.predict(obs, deterministic=True)

        # 환경에 행동 적용
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        # 에피소드 종료 조건
        done = terminated or truncated
        if done:
            break

    return total_reward, steps


def main():
    print("=" * 50)
    print("CartPole 테스트 시작")
    print(f"  모델 경로: {MODEL_PATH}")
    print(f"  렌더링: {RENDER}")
    print("  (best_model.zip이 업데이트되면 자동으로 재로드합니다)")
    print("=" * 50)

    # 환경 생성 (render_mode="human"이면 화면에 시각화)
    render_mode = "human" if RENDER else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # 초기 모델 설정
    # best_model.zip이 없으면 랜덤 초기화 모델로 시작
    if os.path.exists(MODEL_PATH):
        model, last_mtime = load_model(MODEL_PATH)
    else:
        print("\nbest_model.zip이 없습니다. 랜덤 초기화 모델로 시작합니다.")
        print("  train.py를 실행하면 학습이 진행되면서 모델이 자동으로 업데이트됩니다.\n")
        # 학습되지 않은 초기 상태의 PPO 모델 생성
        model = PPO("MlpPolicy", env)
        last_mtime = None  # 아직 파일이 없으므로 None으로 초기화

    episode = 0
    last_check_time = time.time()

    print("\n테스트를 시작합니다. Ctrl+C로 종료하세요.\n")

    try:
        while True:
            episode += 1

            # ─────────────────────────────────────────────
            # 모델 변경 감지 (실시간 업데이트)
            # ─────────────────────────────────────────────
            # MODEL_CHECK_INTERVAL마다 파일 수정 시간을 확인
            now = time.time()
            if now - last_check_time >= MODEL_CHECK_INTERVAL:
                last_check_time = now
                try:
                    current_mtime = os.path.getmtime(MODEL_PATH)
                    if current_mtime != last_mtime:
                        print("\n" + "=" * 50)
                        if last_mtime is None:
                            # 랜덤 모델로 플레이 중에 처음으로 best_model.zip 생성됨
                            print("  [모델 전환] 학습된 Best 모델로 전환합니다!")
                        else:
                            # 기존 best_model.zip이 더 좋은 모델로 갱신됨
                            print("  [모델 업데이트] 더 좋은 Best 모델로 갱신되었습니다!")
                        print(f"  업데이트 시각 : {time.strftime('%H:%M:%S')}")
                        print(f"  에피소드      : {episode}")
                        print("=" * 50)
                        model, last_mtime = load_model(MODEL_PATH)
                        print()
                except OSError:
                    # 파일이 쓰여지는 도중일 수 있음, 잠시 후 재시도
                    pass

            # 에피소드 실행
            total_reward, steps = run_episode(env, model, render=RENDER)

            print(f"  에피소드 {episode:4d} | 보상: {total_reward:6.1f} | 스텝: {steps:4d}")

            # 에피소드 간 잠시 대기
            if RENDER:
                time.sleep(EPISODE_DELAY)

    except KeyboardInterrupt:
        print("\n\n테스트 종료.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
