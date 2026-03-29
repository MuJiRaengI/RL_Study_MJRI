"""
study3/callbacks.py - 커스텀 학습 콜백
=========================================
Walk the Stork 학습에 사용하는 SB3 커스텀 콜백을 정의합니다.

BestModelCallback:
  - study1/2의 EvalCallback이 하는 best_model.zip 저장 역할을 대신합니다.
  - 실제 게임 환경은 별도 eval 환경을 만들 수 없으므로
    학습 중 에피소드 보상을 직접 추적하여 최고 모델을 저장합니다.
"""

import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class BestModelCallback(BaseCallback):
    """
    학습 중 최고 성능 모델을 자동 저장하는 콜백.

    동작 방식:
      - 에피소드가 끝날 때마다 보상을 기록합니다.
      - 최근 window개 에피소드의 평균 보상이 역대 최고를 갱신하면
        best_model.zip을 저장합니다.

    EvalCallback 대신 사용하는 이유:
      - EvalCallback은 별도의 평가용 환경이 필요합니다.
      - 실제 게임은 환경이 1개뿐이므로, 학습 도중 별도 평가가 불가합니다.
    """

    def __init__(self, save_path: str, window: int = 5, verbose: int = 1):
        """
        Args:
            save_path: best_model.zip을 저장할 디렉토리 경로
            window: 평균 보상 계산에 사용할 최근 에피소드 수
            verbose: 1이면 갱신 시 출력
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.window = window
        self.episode_rewards = []               # 에피소드별 보상 누적 기록
        self.best_mean_reward = -float("inf")   # 현재까지 최고 평균 보상

    def _on_step(self) -> bool:
        """
        매 스텝마다 호출됩니다.

        EpisodeBoundaryPPO의 learn()이 callback.update_locals(locals())를 호출하여
        infos를 self.locals에 전달합니다. 에피소드 종료 여부는 infos로 확인합니다.
        """
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # 에피소드 종료: 원본 보상(VecNormalize 정규화 이전)을 기록
                episode_reward = info["episode"]["r"]
                self.episode_rewards.append(episode_reward)

                # window개 이상 쌓이면 평균 계산 후 최고 모델 갱신 여부 판단
                if len(self.episode_rewards) >= self.window:
                    mean_reward = float(np.mean(self.episode_rewards[-self.window:]))

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # best_model.zip 저장
                        self.model.save(os.path.join(self.save_path, "best_model"))
                        if self.verbose > 0:
                            print(
                                f"\n  [BestModel] 최고 모델 갱신! "
                                f"평균 보상: {mean_reward:.2f} "
                                f"(누적 에피소드: {len(self.episode_rewards)})"
                            )
        return True
