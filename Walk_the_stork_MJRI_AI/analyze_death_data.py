import os
import pickle
import numpy as np
import cv2
from pathlib import Path


def visualize_death_data(pkl_path, save_dir):
    """
    죽기 2초 전 데이터를 시각화하여 이미지로 저장

    Args:
        pkl_path: pickle 파일 경로
        save_dir: 이미지 저장 디렉토리
    """
    # pickle 파일 로드
    if not os.path.exists(pkl_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pkl_path}")
        return

    print(f"📂 데이터 로드 중: {pkl_path}")
    with open(pkl_path, "rb") as f:
        death_data = pickle.load(f)

    print(f"✅ {len(death_data)}개 프레임 로드 완료")

    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # action 이름 매핑
    action_names = {0: "LEFT", 1: "STOP", 2: "RIGHT"}

    # 각 프레임 처리
    for idx, frame_data in enumerate(death_data):
        obs = frame_data["obs"]
        action = frame_data["action"]
        step = frame_data["step"]

        # VecEnv 배열에서 실제 obs 추출 (shape: (1, H, W, C))
        if obs.shape[0] == 1:
            obs = obs[0]  # (H, W, C)

        # action도 배열이면 첫 번째 요소 추출
        if isinstance(action, np.ndarray) and action.size > 0:
            action_idx = action[0] if action.ndim > 0 else action
        else:
            action_idx = action

        action_name = action_names.get(int(action_idx), f"UNKNOWN({action_idx})")

        # 마지막 채널만 사용 (최신 프레임)
        # obs shape: (H, W, channels*stacked_num)
        # 마지막 채널 추출
        if obs.shape[-1] > 1:
            last_frame = obs[:, :, -1]  # 마지막 채널 (최신 프레임)
        else:
            last_frame = obs[:, :, 0]

        # grayscale이면 BGR로 변환 (텍스트 색상을 위해)
        if len(last_frame.shape) == 2:
            img = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
        else:
            img = last_frame.copy()

        # 이미지 크기 확인
        h, w = img.shape[:2]

        # 텍스트 추가를 위한 상단 여백 생성 (40픽셀)
        margin = 40
        img_with_text = np.zeros((h + margin, w, 3), dtype=np.uint8)
        img_with_text[margin:, :] = img

        # 상단 배경을 흰색으로
        img_with_text[:margin, :] = 255

        # 텍스트 추가
        text = f"Step: {step} | Action: {action_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 0, 0)  # 검은색

        # 텍스트 위치 (중앙 정렬)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (margin + text_size[1]) // 2

        cv2.putText(
            img_with_text,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

        # 이미지 저장
        save_path = os.path.join(save_dir, f"frame_{idx:03d}_step_{step}.png")
        cv2.imwrite(save_path, img_with_text)

        if (idx + 1) % 10 == 0:
            print(f"  처리 중... {idx + 1}/{len(death_data)}")

    print(f"✅ 모든 프레임 저장 완료: {save_dir}")
    print(f"📊 총 {len(death_data)}개 이미지 생성")


def main():
    """메인 함수"""
    print("=" * 60)
    print("죽기 2초 전 데이터 시각화 도구")
    print("=" * 60)

    # 기본 경로 설정 (원하는 대로 수정 가능)
    default_pkl_path = "./death_data/episode_2_death_data.pkl"
    default_save_dir = "./death_frames/"

    print()
    visualize_death_data(default_pkl_path, default_save_dir)


if __name__ == "__main__":
    main()
