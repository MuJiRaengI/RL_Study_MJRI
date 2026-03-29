import torch
import numpy as np
import gymnasium as gym
import ale_py
import time
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from typing import Optional, Union
import warnings

# 필수: Atari 환경 등록
gym.register_envs(ale_py)


class MJRIPPO(PPO):
    @classmethod
    def load(
        cls,
        path: str,
        env=None,
        device: Union[torch.device, str] = "auto",
        custom_objects=None,
        print_system_info: bool = False,
        force_reset: bool = True,
        load_partial: bool = True,
        **kwargs,
    ):
        """
        사전학습된 모델을 불러옵니다. action space가 다를 경우 호환 가능한 weight만 불러옵니다.

        Args:
            load_partial: True이면 action space가 달라도 호환 가능한 weight만 불러옴
        """
        if load_partial and env is not None:
            try:
                # 먼저 환경 없이 모델 로드 시도
                loaded_model = super().load(
                    path,
                    env=None,
                    device=device,
                    custom_objects=custom_objects,
                    print_system_info=print_system_info,
                    force_reset=force_reset,
                    **kwargs,
                )

                # action space 체크
                if loaded_model.action_space != env.action_space:
                    warnings.warn(
                        f"⚠️  Action space 불일치 감지! "
                        f"사전학습 모델: {loaded_model.action_space}, "
                        f"현재 환경: {env.action_space}\n"
                        f"→ Policy head를 제외한 weight만 로드합니다."
                    )

                    # 새로운 환경에 맞는 모델 생성
                    new_model = cls(
                        loaded_model.policy_class,
                        env,
                        device=device,
                        _init_setup_model=True,
                        **kwargs,
                    )

                    # State dict 가져오기
                    pretrained_dict = loaded_model.policy.state_dict()
                    new_dict = new_model.policy.state_dict()

                    # 호환 가능한 weight만 필터링 (action_net 제외)
                    filtered_dict = {}
                    skipped_keys = []

                    for k, v in pretrained_dict.items():
                        # action_net과 value_net의 마지막 레이어는 스킵
                        if "action_net" in k or "value_net" in k:
                            if k in new_dict and v.shape == new_dict[k].shape:
                                filtered_dict[k] = v
                            else:
                                skipped_keys.append(k)
                        else:
                            # Feature extractor는 모두 로드
                            if k in new_dict and v.shape == new_dict[k].shape:
                                filtered_dict[k] = v
                            else:
                                skipped_keys.append(k)

                    # 필터링된 weight 로드
                    new_model.policy.load_state_dict(filtered_dict, strict=False)

                    print(
                        f"✅ 호환 가능한 {len(filtered_dict)}/{len(pretrained_dict)} 개의 weight를 로드했습니다."
                    )
                    if skipped_keys:
                        print(f"⏭️  스킵된 레이어: {skipped_keys}")

                    del loaded_model  # 메모리 정리
                    return new_model
                else:
                    # Action space가 같으면 환경 설정만 업데이트
                    loaded_model.set_env(env)
                    return loaded_model

            except Exception as e:
                warnings.warn(f"⚠️  부분 로드 실패: {e}\n기본 로드 방식으로 시도합니다.")
                return super().load(
                    path,
                    env=env,
                    device=device,
                    custom_objects=custom_objects,
                    print_system_info=print_system_info,
                    force_reset=force_reset,
                    **kwargs,
                )
        else:
            # load_partial=False이거나 env가 없으면 기본 동작
            return super().load(
                path,
                env=env,
                device=device,
                custom_objects=custom_objects,
                print_system_info=print_system_info,
                force_reset=force_reset,
                **kwargs,
            )

    def real_learn(
        self,
        total_timesteps: int,
        min_train_steps: int = 2048,
        callback=None,
        log_interval: int = 1,
    ):
        """
        [목표]
        1. 여러 에피소드를 수행하며 데이터를 모은다.
        2. 모인 데이터가 min_train_steps(예: 2048)를 넘으면
        3. '죽은 타이밍'에 학습을 수행한다.
        """

        total_timesteps, callback = self._setup_learn(total_timesteps, callback)
        iteration = 0

        print(
            f"🚀 학습 시작! 데이터가 {min_train_steps} 스텝 이상 모이면, 죽은 뒤 학습합니다."
        )

        while self.num_timesteps < total_timesteps:
            # === [Step 1] Play Loop (데이터 수집) ===
            self.policy.set_training_mode(False)
            self.rollout_buffer.reset()

            # 버퍼가 차면 여러 판을 반복해서 플레이
            while True:
                # 첫 번째 에피소드이거나, 학습 후 재개 시에만 reset
                if self._last_obs is None:
                    self._last_obs = self.env.reset()

                # --- 1판 플레이 (죽을 때까지) ---
                while True:
                    with torch.no_grad():
                        obs_tensor = obs_as_tensor(self._last_obs, self.device)
                        actions, values, log_probs = self.policy(obs_tensor)

                    actions = actions.cpu().numpy()
                    clipped_actions = actions
                    if isinstance(self.action_space, gym.spaces.Box):
                        clipped_actions = np.clip(
                            actions, self.action_space.low, self.action_space.high
                        )

                    new_obs, rewards, dones, infos = self.env.step(clipped_actions)

                    # 🌟 [추가된 부분] 점수와 에피소드 길이 기록 🌟
                    # 이 코드가 없으면 로그에 ep_rew_mean 등이 뜨지 않습니다.
                    self._update_info_buffer(infos)

                    self.num_timesteps += 1

                    # Callback 처리 - locals 업데이트 후 호출
                    if callback:
                        # 콜백이 infos에 접근할 수 있도록 locals 설정
                        callback.update_locals(locals())
                        if not callback.on_step():
                            return self

                    if isinstance(self.action_space, gym.spaces.Discrete):
                        actions = actions.reshape(-1, 1)

                    # 버퍼에 추가
                    if not self.rollout_buffer.full:
                        self.rollout_buffer.add(
                            self._last_obs,
                            actions,
                            rewards,
                            self._last_episode_starts,
                            values,
                            log_probs,
                        )

                    self._last_obs = new_obs
                    self._last_episode_starts = dones

                    # 죽음(Done) 체크
                    if dones[0]:
                        # 죽었을 때 self._last_obs를 업데이트하지만 reset은 하지 않음
                        self._last_obs = new_obs
                        self._last_episode_starts = dones
                        break  # 1판 끝!

                # --- 1판 끝난 후 체크 ---
                # 현재 버퍼에 쌌인 데이터 개수 확인
                current_steps = self.rollout_buffer.pos

                # [핵심 로직] 목표 스텝(2048)보다 많이 모였으면 -> 수집 중단하고 학습하러 감
                print(f"📊 수집된 데이터: {current_steps}/{min_train_steps} 스텝")
                if current_steps >= min_train_steps:
                    print(f"✅ 목표 달성! 학습 시작...")
                    break

                # 아직 부족하면 다음 판 시작 (VecEnv가 이미 reset했으므로 new_obs 사용)

            # 여기서 ESC를 눌러서 게임 잠시 멈춤
            print("⏸️  학습 시작 전 게임 일시정지...")
            self.env.env_method("perform", "pause")
            time.sleep(0.2)

            # === [Step 2] Train Loop (학습) ===

            # 🔔 Rollout 종료 콜백 호출 (distance 로깅용)
            if callback:
                callback.on_rollout_end()

            with torch.no_grad():
                obs_tensor = obs_as_tensor(new_obs, self.device)
                last_values = self.policy.predict_values(obs_tensor)

            # 버퍼 크기 해킹 (AssertionError 방지 & 유효 데이터만 학습)
            real_buffer_size = self.rollout_buffer.buffer_size
            self.rollout_buffer.buffer_size = (
                self.rollout_buffer.pos
            )  # 현재 모인 개수(예: 2200)로 설정
            self.rollout_buffer.full = True  # 꽉 찼다고 속임

            # GAE 계산 및 학습
            self.rollout_buffer.compute_returns_and_advantage(
                last_values=last_values, dones=dones
            )
            self.train()  # 학습!

            # 버퍼 원상복구
            self.rollout_buffer.buffer_size = real_buffer_size
            self.rollout_buffer.full = False

            # 🔄 학습 후 환경 리셋 준비 (다음 루프에서 reset 호출됨)
            self._last_obs = None
            self._last_episode_starts = None

            iteration += 1
            print(
                f"✅ [{iteration}/{log_interval}]학습 완료! 다시 플레이로 돌아갑니다."
            )
            # 매 iteration마다 로그 dump (콜백의 distance 로깅 포함)
            self._dump_logs()

            # 여기서 스페이스바를 눌러 게임 다시 시작
            print("▶️  학습 완료 후 게임 재개...")
            self.env.env_method("perform", "continue")
            self.env.env_method("perform", "continue")
            time.sleep(0.5)

        return self
