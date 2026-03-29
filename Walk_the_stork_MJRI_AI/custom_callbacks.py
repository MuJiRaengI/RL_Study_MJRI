from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import os


class DistanceLoggingCallback(BaseCallback):
    """
    에피소드 최대 거리를 로그에 기록하는 커스텀 콜백
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_distances = []

    def _on_step(self) -> bool:
        # info에서 episode 정보 확인
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info and "distance" in info:
                distance = info["distance"]
                self.episode_distances.append(distance)

        return True  # 계속 학습 진행

    def _on_rollout_end(self) -> None:
        """롤아웃 종료 시 바로 로깅하고 버퍼 초기화"""
        if len(self.episode_distances) > 0:
            mean_distance = np.mean(self.episode_distances)
            max_distance = np.max(self.episode_distances)

            # TensorBoard에 로그
            self.logger.record("rollout/ep_distance_mean", mean_distance)
            self.logger.record("rollout/ep_distance_max", max_distance)

            print(
                f"📊 [Distance] 평균: {mean_distance:.2f}m, 최대: {max_distance:.2f}m"
            )

            # 현재 버퍼 초기화
            self.episode_distances = []


class VecNormalizeCheckpointCallback(CheckpointCallback):
    """CheckpointCallback을 확장하여 VecNormalize도 함께 저장"""

    def _on_step(self) -> bool:
        result = super()._on_step()

        # VecNormalize 저장
        if self.n_calls % self.save_freq == 0:
            if isinstance(self.model.get_env(), VecNormalize):
                vec_normalize_path = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl",
                )
                self.model.get_env().save(vec_normalize_path)
                if self.verbose > 0:
                    print(f"VecNormalize 저장: {vec_normalize_path}")

        return result
