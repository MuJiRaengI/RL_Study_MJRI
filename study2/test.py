"""
study2/test.py - Breakout 테스트 코드
======================================
train.py와 동시에 실행하면 학습 과정을 실시간으로 확인할 수 있습니다.

핵심 기능:
  - best_model.zip이 없으면 랜덤 초기화 모델로 시작 (학습 전 상태)
  - best_model.zip이 생기거나 업데이트되면 자동으로 재로드
  - 에피소드별 점수 출력
  - 렌더링 ON/OFF 선택 가능

사용법:
  1. train.py와 test.py를 동시에 실행하세요
  2. 처음에는 랜덤 행동으로 점수가 낮다가, 학습이 진행될수록 점점 점수가 올라갑니다
"""

import os
import time
import numpy as np
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Atari 환경을 gymnasium에 등록 (없으면 BreakoutNoFrameskip 등을 찾지 못함)
gym.register_envs(ale_py)

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────

# 이 스크립트 파일 위치 기준으로 경로 설정 (어디서 실행해도 항상 study2/models/에서 로드)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "best_model.zip")
ENV_ID = "BreakoutNoFrameskip-v4"
N_STACK = 4  # train.py와 동일하게 맞춰야 함

# 렌더링 여부
RENDER = True

# 모델 파일 변경 감지 주기 (초)
MODEL_CHECK_INTERVAL = 2.0


def load_model(model_path: str) -> tuple:
    """모델을 로드하고 수정 시간을 반환합니다."""
    print(f"  모델 로드 중: {model_path}")
    model = PPO.load(model_path)
    mtime = os.path.getmtime(model_path)
    print(f"  모델 로드 완료 (수정 시간: {time.ctime(mtime)})")
    return model, mtime


def run_episode(vec_env, model: PPO) -> float:
    """
    에피소드를 한 번 실행하고 총 보상을 반환합니다.
    VecEnv를 사용하므로 관측값은 배열 형태입니다.
    """
    obs = vec_env.reset()
    total_reward = 0.0

    while True:
        # 행동 선택
        action, _ = model.predict(obs, deterministic=True)

        # 환경 진행
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += rewards[0]  # 1개 환경이므로 index 0

        # VecEnv에서는 에피소드 종료 후 자동으로 reset됨
        # dones[0]이 True면 에피소드 종료
        if dones[0]:
            break

    return total_reward


def main():
    print("=" * 50)
    print("Breakout 테스트 시작")
    print(f"  모델 경로: {MODEL_PATH}")
    print(f"  렌더링: {RENDER}")
    print("  (best_model.zip이 업데이트되면 자동으로 재로드합니다)")
    print("=" * 50)

    # 테스트 환경 생성
    # render_mode는 make_atari_env의 env_kwargs로 전달
    render_mode = "human" if RENDER else "rgb_array"
    test_env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=0,
        env_kwargs={"render_mode": render_mode}
    )
    test_env = VecFrameStack(test_env, n_stack=N_STACK)

    # 초기 모델 설정
    # best_model.zip이 없으면 랜덤 초기화 모델로 시작
    if os.path.exists(MODEL_PATH):
        model, last_mtime = load_model(MODEL_PATH)
    else:
        print("\nbest_model.zip이 없습니다. 랜덤 초기화 모델로 시작합니다.")
        print("  train.py를 실행하면 학습이 진행되면서 모델이 자동으로 업데이트됩니다.\n")
        # 학습되지 않은 초기 상태의 PPO 모델 생성
        model = PPO("CnnPolicy", test_env)
        last_mtime = None  # 아직 파일이 없으므로 None으로 초기화

    episode = 0
    scores = []
    last_check_time = time.time()

    print("\n테스트를 시작합니다. Ctrl+C로 종료하세요.\n")

    try:
        while True:
            episode += 1

            # ─────────────────────────────────────────────
            # 모델 변경 감지 (실시간 업데이트)
            # ─────────────────────────────────────────────
            now = time.time()
            if now - last_check_time >= MODEL_CHECK_INTERVAL:
                last_check_time = now
                try:
                    current_mtime = os.path.getmtime(MODEL_PATH)
                    if current_mtime != last_mtime:
                        print("\n" + "=" * 50)
                        if last_mtime is None:
                            print("  [모델 전환] 학습된 Best 모델로 전환합니다!")
                        else:
                            print("  [모델 업데이트] 더 좋은 Best 모델로 갱신되었습니다!")
                        print(f"  업데이트 시각 : {time.strftime('%H:%M:%S')}")
                        print(f"  에피소드      : {episode}")
                        print(f"  최근 10 평균  : {np.mean(scores[-10:]):.1f}" if scores else "")
                        print("=" * 50)
                        model, last_mtime = load_model(MODEL_PATH)
                        print()
                except OSError:
                    pass

            # 에피소드 실행
            score = run_episode(test_env, model)
            scores.append(score)

            # 최근 10 에피소드 평균 계산
            recent_avg = np.mean(scores[-10:])
            print(
                f"  에피소드 {episode:4d} | "
                f"점수: {score:6.1f} | "
                f"최근 10 평균: {recent_avg:6.1f}"
            )

    except KeyboardInterrupt:
        print("\n\n테스트 종료.")
        if scores:
            print(f"  전체 평균 점수: {np.mean(scores):.1f}")
            print(f"  최고 점수: {max(scores):.1f}")

    finally:
        test_env.close()


if __name__ == "__main__":
    main()
