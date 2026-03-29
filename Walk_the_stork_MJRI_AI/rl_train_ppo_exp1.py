import torch
import os  # 폴더 생성을 위해 추가
from typing import Callable
from ppo import MJRIPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from env import WalkTheStork

from custom_callbacks import DistanceLoggingCallback

from consts import *


def train():
    # ==========================================
    # 📂 [설정] 저장 경로 관리
    # ==========================================
    # 원하는 폴더 이름을 여기에 지정하세요.
    save_dir = "./ppo_exp2/"

    # 폴더가 없으면 에러 없이 자동으로 생성합니다.
    os.makedirs(save_dir, exist_ok=True)

    print(f"📂 모든 모델과 체크포인트는 '{save_dir}' 폴더에 저장됩니다.")
    # ==========================================

    # 1. 환경 설정 z
    def make_env():
        env = WalkTheStork(
            fps=30,
            crop_pos=(100, 300, 1050, 750),
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
    # VecNormalize로 입력과 reward normalize
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    model = MJRIPPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        tensorboard_log=os.path.join(save_dir, "custom_ppo_logs"),
        # --- [표준 하이퍼파라미터 적용 구간] ---
        learning_rate=3e-6,
        n_steps=16384,
        batch_size=64,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    )

    pretrained_path = None
    pretrained_path = r"C:\Users\stpe9\Desktop\vscode\Walk_the_stork_MJRI_AI\ppo_exp1\ppo_Geometry_1300000_steps.zip"
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"🔄 사전학습된 모델 불러오는 중: {pretrained_path}")
        # 환경 없이 로드해서 weight만 가져오기
        pretrained_model = MJRIPPO.load(pretrained_path, env=None, device="cuda:0")

        # policy weight를 현재 모델로 복사 (strict=False로 shape 불일치 무시)
        model.policy.load_state_dict(pretrained_model.policy.state_dict(), strict=False)

        # 로드된 weight 확인
        loaded_keys = len(pretrained_model.policy.state_dict())
        current_keys = len(model.policy.state_dict())
        print(
            f"✅ 사전학습된 모델의 weight 로드 완료! ({loaded_keys}/{current_keys} keys)"
        )
        del pretrained_model  # 메모리 절약

    # [수정] 체크포인트 저장 경로를 save_dir로 통일
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=save_dir,
        name_prefix="ppo_Geometry",
    )

    distance_callback = DistanceLoggingCallback(verbose=1)

    # 여러 콜백을 함께 사용
    callback_list = CallbackList([checkpoint_callback, distance_callback])

    print("🚀 Custom PPO 학습 시작... ")

    model.real_learn(
        total_timesteps=10_000_000,
        log_interval=1,
        min_train_steps=2048,
        callback=callback_list,
    )

    # [수정] 최종 모델도 같은 폴더에 저 장 (경로 결합: os.path.join)
    final_model_path = os.path.join(save_dir, "ppo_final")
    model.save(final_model_path)

    print(f"✅ 학습 완료! 최종 모델 저장 경로: {final_model_path}.zip")


if __name__ == "__main__":
    train()
