import torch
import os
import numpy as np
import time
from ppo import MJRIPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env import WalkTheStork

from consts import *


def evaluate():
    # ==========================================
    # 📂 [설정] 모델 경로
    # ==========================================
    model_path = r"C:\Users\stpe9\Desktop\vscode\Walk_the_stork_MJRI_AI\ppo_exp3_2\ppo_Geometry_1850000_steps.zip"

    if not os.path.exists(model_path):
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return

    print(f"📂 모델 로드 중: {model_path}")
    # ==========================================

    # 1. 환경 설정 (학습과 동일하게 VecNormalize 적용)
    def make_env():
        env = WalkTheStork(
            fps=30,
            crop_pos=(300, 600, 700, 450),
            resize=(320, 180),
            action_num=3,
            stacked_num=4,
            gray_scale=True,
            device="cuda:0",
        )
        env.run_capture()
        return env

    # 단일 환경을 VecEnv로 감싸기
    env = DummyVecEnv([make_env])
    # VecNormalize로 입력과 reward normalize (학습 시와 동일하게)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    # 2. 학습된 모델 로드
    model = MJRIPPO.load(model_path, env=env)
    print(f"✅ 모델 로드 완료!")

    # 3. 평가 설정
    n_eval_episodes = 100  # 평가할 에피소드 수
    episode_rewards = []
    episode_lengths = []
    episode_distances = []

    print(f"\n🎮 {n_eval_episodes}개 에피소드 평가 시작...\n")

    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = [False]

        while not done[0]:
            # 모델로 행동 예측 (deterministic=True로 탐험 없이 최선의 행동 선택)
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]  # VecEnv는 배열로 반환
            episode_length += 1

            # 에피소드 종료 시 거리 정보 저장
            if done[0]:
                if len(info) > 0 and "distance" in info[0]:
                    episode_distances.append(info[0]["distance"])

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        distance_str = (
            f", Distance = {episode_distances[-1]:.2f}m" if episode_distances else ""
        )
        print(
            f"Episode {episode + 1}/{n_eval_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}{distance_str}"
        )
        time.sleep(2.0)  # 출력이 너무 빠르게 지나가지 않도록 약간의 지연 추가

    # 4. 통계 출력
    print("\n" + "=" * 50)
    print("📊 평가 결과")
    print("=" * 50)
    print(f"평균 보상: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"평균 길이: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    if episode_distances:
        print(
            f"평균 거리: {np.mean(episode_distances):.2f}m ± {np.std(episode_distances):.2f}m"
        )
        print(f"최대 거리: {np.max(episode_distances):.2f}m")
    print(f"최고 보상: {np.max(episode_rewards):.2f}")
    print(f"최저 보상: {np.min(episode_rewards):.2f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
