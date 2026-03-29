"""
study3/ppo.py - 에피소드 경계 PPO
====================================
실제 게임 환경을 위한 커스텀 PPO 클래스를 정의합니다.

표준 PPO와의 차이:
  표준 PPO: n_steps만큼 데이터가 모이면 즉시 학습 시작
            → 에피소드 중간에 학습이 시작될 수 있음 (실제 게임에서 부적합)

  EpisodeBoundaryPPO: 에피소드(게임 한 판)가 끝날 때마다 데이터 수를 확인
                      → min_train_steps 이상이면 그 에피소드 종료 후 학습 시작
                      → 항상 에피소드 경계에서만 학습을 수행
"""

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor


class EpisodeBoundaryPPO(PPO):
    """
    에피소드 경계에서만 학습하는 커스텀 PPO.

    SB3의 PPO를 상속하여 learn() 메서드만 오버라이드합니다.
    policy, 하이퍼파라미터, 저장/로드 등 나머지는 모두 표준 PPO와 동일합니다.

    동작 방식:
      [수집 단계]
        1. 에피소드를 플레이하며 rollout buffer에 데이터를 쌓습니다.
        2. 에피소드가 끝날 때마다 buffer에 쌓인 데이터 수를 확인합니다.
        3. min_train_steps 이상 쌓이면 수집을 종료합니다.
           (에피소드 중간에 끊지 않고 반드시 에피소드가 끝난 후에 종료)
      [학습 단계]
        4. GAE(Generalized Advantage Estimation)를 계산합니다.
        5. 수집된 데이터로 PPO 그래디언트 업데이트를 수행합니다.
        6. 다음 에피소드를 위해 환경을 reset하고 1단계로 돌아갑니다.
    """

    def learn(
        self,
        total_timesteps: int,
        min_train_steps: int = 2048,
        callback=None,
        log_interval: int = 1,
        tb_log_name: str = "EpisodeBoundaryPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        에피소드 경계에서만 학습하는 커스텀 학습 루프.

        Args:
            total_timesteps: 총 학습 스텝 수
            min_train_steps: 학습 시작 조건 (에피소드 종료 후 이 값 이상이면 학습)
            callback: SB3 콜백 (BestModelCallback, CheckpointCallback 등)
        """
        # SB3 내부 설정 초기화 (로거, 콜백, 랜덤 시드 등)
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

        iteration = 0
        print(
            f"\n학습 시작! "
            f"데이터가 {min_train_steps}스텝 이상 모이면, 에피소드 종료 후 학습합니다."
        )

        while self.num_timesteps < total_timesteps:
            # ─────────────────────────────────────────────
            # [수집 단계] 에피소드 단위로 데이터 수집
            # ─────────────────────────────────────────────
            self.policy.set_training_mode(False)  # 탐험 모드 (dropout 등 비활성화)
            self.rollout_buffer.reset()             # 버퍼 초기화

            # 처음이거나 학습 후 재개 시: 환경 reset
            if self._last_obs is None:
                self._last_obs = self.env.reset()
                # 새 에피소드 시작을 나타내는 플래그
                self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

            # min_train_steps 이상 모일 때까지 에피소드를 반복
            while True:
                # --- 에피소드 한 판 플레이 ---
                while True:
                    with torch.no_grad():
                        # 현재 관측값으로 액션, 가치, 로그 확률 계산
                        obs_tensor = obs_as_tensor(self._last_obs, self.device)
                        actions, values, log_probs = self.policy(obs_tensor)

                    actions = actions.cpu().numpy()

                    # 환경에 액션 적용
                    new_obs, rewards, dones, infos = self.env.step(actions)

                    # 에피소드 보상/길이 로깅 (ep_rew_mean 출력에 필요)
                    self._update_info_buffer(infos)
                    self.num_timesteps += 1

                    # 콜백 호출 (BestModelCallback 등)
                    callback.update_locals(locals())
                    if not callback.on_step():
                        return self  # 콜백이 False를 반환하면 학습 조기 종료

                    # Discrete 액션은 (n_envs, 1) 형태로 변환 후 버퍼에 추가
                    buffer_actions = actions.reshape(-1, 1)

                    # 버퍼가 가득 차지 않았을 때만 추가
                    if not self.rollout_buffer.full:
                        self.rollout_buffer.add(
                            self._last_obs,
                            buffer_actions,
                            rewards,
                            self._last_episode_starts,
                            values,
                            log_probs,
                        )

                    self._last_obs = new_obs
                    self._last_episode_starts = dones  # done=True면 다음 관측은 새 에피소드 시작

                    # 에피소드 종료 (사망)
                    if dones[0]:
                        break  # 한 판 종료 → 외부 while에서 버퍼 체크

                # 에피소드 종료 후 버퍼 크기 확인
                current_steps = self.rollout_buffer.pos
                print(f"  수집된 데이터: {current_steps:,} / {min_train_steps:,} 스텝")

                if current_steps >= min_train_steps:
                    print(f"  목표 달성! 학습을 시작합니다...")
                    break
                # 아직 부족하면 다음 에피소드 계속
                # (DummyVecEnv가 done 시 자동으로 reset하므로 _last_obs는 새 에피소드의 첫 관측)

            # ─────────────────────────────────────────────
            # [학습 단계] GAE 계산 후 PPO 그래디언트 업데이트
            # ─────────────────────────────────────────────
            callback.on_rollout_end()

            with torch.no_grad():
                # 마지막 관측값의 가치(V)를 계산하여 GAE에 사용
                last_obs_tensor = obs_as_tensor(new_obs, self.device)
                last_values = self.policy.predict_values(last_obs_tensor)

            # 버퍼 크기 조정:
            # SB3의 rollout_buffer는 고정 크기(n_steps)를 기대하지만,
            # 에피소드 단위로 수집하면 실제 데이터 수가 n_steps와 다를 수 있습니다.
            # buffer_size를 실제 수집된 수로 일시적으로 변경하여 AssertionError 방지
            real_buffer_size = self.rollout_buffer.buffer_size
            self.rollout_buffer.buffer_size = self.rollout_buffer.pos
            self.rollout_buffer.full = True  # 꽉 찼다고 표시 (GAE 계산 트리거)

            # GAE(Generalized Advantage Estimation) 계산
            self.rollout_buffer.compute_returns_and_advantage(
                last_values=last_values, dones=dones
            )

            # PPO 그래디언트 업데이트
            self.train()

            # 버퍼 크기 복구
            self.rollout_buffer.buffer_size = real_buffer_size
            self.rollout_buffer.full = False

            # 다음 수집 단계를 위해 reset 준비
            # (None으로 설정하면 다음 루프에서 env.reset() 호출)
            self._last_obs = None
            self._last_episode_starts = None

            iteration += 1
            print(f"  [{iteration}번째 학습 완료] 다음 에피소드를 시작합니다.")

            # 로그 출력 (ep_rew_mean 등 SB3 내장 통계)
            self._dump_logs()

        return self
