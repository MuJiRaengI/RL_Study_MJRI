"""
study3/custom_env.py - Walk the Stork 커스텀 환경
==================================================
Chrome 브라우저에서 실행 중인 Flash 게임 "황새 오래 걷기"를
화면 캡처와 키보드 입력으로 제어하는 gymnasium 커스텀 환경입니다.

Walk the Stork란?
  - 황새 캐릭터를 좌우로 움직여 최대한 오래 걷게 하는 게임
  - 관측값(observation): 게임 화면 이미지 (180 x 320 x 4, 그레이스케일 4프레임 스택)
  - 행동(action): 0 = 왼쪽, 1 = 정지, 2 = 오른쪽
  - 보상(reward): 화면 내 위치 기반 (위쪽일수록 좋음)
      - 생존 시: +0.5 - (화면 위에서 황새 위치 * 0.01)
      - 사망 시: -500 + 진행 거리 (m)
  - 종료 조건: 스택된 4프레임이 모두 동일하면 사망으로 판단

사전 준비:
  1. Chrome에서 '황새 오래 걷기' 게임을 실행하세요
  2. 창 제목이 WINDOW_TITLE 상수와 정확히 일치해야 합니다
  3. 필요 패키지: pywin32 mss easyocr opencv-python

사용 방법:
  env = WalkTheStorkEnv()
  env.run_capture()       ← 반드시 호출해야 화면 캡처가 시작됩니다
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action)
  env.close()
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import win32gui
import win32con
import win32api
import time
import cv2
import threading
import torch
from collections import deque
import easyocr
import re

from capture import GameCapture
from consts import LEFT_ARROW, STOP, RIGHT_ARROW, DIE, LIVE, UNKNOWN


# ─────────────────────────────────────────────
# 창 제목 상수 (변경 시 여기만 수정)
# ─────────────────────────────────────────────
WINDOW_TITLE = "황새 오래 걷기 (Walk the Stork) - 플래시게임 | 와플래시 게임 아카이브 - Chrome"


class WalkTheStorkEnv(gym.Env):
    """
    Walk the Stork 게임을 제어하는 gymnasium 환경.

    화면 캡처(mss)와 Windows API(win32api)를 이용해
    실제 게임을 관측하고 키보드 입력으로 제어합니다.
    """

    def __init__(
        self,
        fps: int = 30,
        crop_pos: tuple = (300, 600, 700, 450),
        resize: tuple = (320, 180),
        action_num: int = 3,
        stacked_num: int = 4,
        gray_scale: bool = True,
        device: str = "cuda:0",
    ):
        """
        Args:
            fps (int): 목표 캡처 및 스텝 FPS
            crop_pos (tuple): 화면 크롭 영역 (x, y, width, height)
            resize (tuple): 크롭 후 리사이즈 크기 (width, height)
            action_num (int): 액션 수 (기본 3: 왼쪽/정지/오른쪽)
            stacked_num (int): 스택할 프레임 수 (기본 4)
            gray_scale (bool): True이면 그레이스케일로 변환
            device (str): 모델 디바이스 (CUDA 없으면 자동으로 CPU 사용)
        """
        super().__init__()

        self.window_title = WINDOW_TITLE
        self.crop_pos = crop_pos
        self.resize = resize
        self.fps = fps
        self.stacked_num = stacked_num
        self.gray_scale = gray_scale
        self.device = device if torch.cuda.is_available() else "cpu"

        # ─────────────────────────────────────────────
        # 액션 공간: Discrete(3) → 0=왼쪽, 1=정지, 2=오른쪽
        # ─────────────────────────────────────────────
        self.action_space = spaces.Discrete(action_num)

        # ─────────────────────────────────────────────
        # 관측 공간: 그레이스케일 프레임 stacked_num개를 채널로 스택
        #   gray_scale=True → 채널 수 = 1 * stacked_num = 4
        #   shape: (height, width, channels)  예) (180, 320, 4)
        # ─────────────────────────────────────────────
        ch = 1 if gray_scale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(resize[1], resize[0], ch * stacked_num),
            dtype=np.uint8,
        )

        # ─────────────────────────────────────────────
        # EasyOCR: 사망 시 화면에 표시된 진행 거리(m)를 읽기 위해 사용
        # 첫 실행 시 모델 다운로드로 약 100MB 네트워크 트래픽이 발생할 수 있습니다
        # ─────────────────────────────────────────────
        print("EasyOCR 초기화 중... (처음 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
        self.ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("EasyOCR 초기화 완료!")

        # 키보드/마우스 입력 딜레이
        self.key_delay = 0.05
        self.mouse_delay = 0.05

        # 화면 캡처 객체 (start는 run_capture()로 별도 호출)
        self.capture = GameCapture(
            self.window_title,
            self.fps,
            self.stacked_num,
            self.crop_pos,
            self.resize,
            self.gray_scale,
        )

        # 에피소드 추적 변수
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_distance = 0.0
        self.current_state = None

        # FPS 기반 스텝 타이밍 제어
        self.frame_interval = 1.0 / self.fps
        self.next_step_time = None

        # 스텝 FPS 측정 (100프레임 이동 평균)
        self.step_frame_times = deque(maxlen=100)
        self.current_step_fps = 0.0
        self.last_step_time = None

        # 마지막으로 계산된 min_h (황새의 화면 상단 기준 위치)
        self.last_min_h = 0

    def run_capture(self):
        """
        화면 캡처 스레드를 시작합니다.

        환경 생성 후 반드시 호출해야 합니다.
        1초 대기 후 반환합니다 (버퍼 안정화 목적).
        """
        self.capture.start()
        time.sleep(1.0)

    def close(self):
        """
        환경을 종료하고 리소스를 정리합니다.

        캡처 스레드를 중지하고 모든 키 입력을 해제합니다.
        """
        # 모든 키 입력 해제
        win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)
        # 캡처 스레드 중지
        if self.capture.running:
            self.capture.stop()

    def get_stacked_buffer(self):
        """
        현재 스택된 프레임을 가져오고 이진화(검은색 추출)를 적용합니다.

        황새와 배경을 구분하기 위해 어두운 픽셀(< 30)을 흰색으로 변환합니다.

        Returns:
            numpy.ndarray 또는 None: (H, W, C*stacked_num) 형태의 배열
        """
        frames = self.capture.get_stacked_frames()
        if frames is None:
            return None

        # 이진화: threshold=30 이하를 255(흰색)으로, 나머지는 0(검정)으로
        # 황새의 검은 실루엣을 흰색으로 변환하여 배경과 구분
        threshold_value = 30
        frames_black = (frames < threshold_value).astype(np.uint8) * 255

        return frames_black

    # ─────────────────────────────────────────────
    # 게임 조작 메서드
    # ─────────────────────────────────────────────

    def perform(self, task: str, delay: float = None):
        """
        게임 조작 태스크를 수행합니다.

        Args:
            task: "press_space", "f5", "click_screen", "pause", "continue"
        """
        if task in ["press_space", "f5"]:
            if delay is None:
                delay = self.key_delay
            self._key(task, delay)
        elif task in ["click_screen"]:
            self._click(task)
        elif task in ["pause", "continue"]:
            # 현재 미구현 (no-op)
            pass
        else:
            raise ValueError(f"알 수 없는 태스크: {task}")

    def _focus(self):
        """게임 창에 포커스를 맞춥니다."""
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.01)  # 포커스 안정화
        except Exception:
            print('focus 실패')
            pass

    def _key(self, task: str, delay: float = None):
        """키보드 입력 시뮬레이션 (스페이스바, F5)."""
        self._focus()

        if task == "press_space":
            win32api.keybd_event(win32con.VK_SPACE, 0, 0, 0)
            time.sleep(self.key_delay)
            win32api.keybd_event(win32con.VK_SPACE, 0, win32con.KEYEVENTF_KEYUP, 0)
        elif task == "f5":
            print("F5 누름 (페이지 새로고침)")
            win32api.keybd_event(win32con.VK_F5, 0, 0, 0)
            time.sleep(self.key_delay)
            win32api.keybd_event(win32con.VK_F5, 0, win32con.KEYEVENTF_KEYUP, 0)
        else:
            raise ValueError(f"알 수 없는 키: {task}")

        if delay is not None and delay > 0:
            time.sleep(delay)

    def _click(self, task: str):
        """창 내 특정 좌표를 클릭합니다 (게임 시작 버튼 클릭 등)."""
        x, y, w, h = self._get_window_rect()
        if task == "click_screen":
            click_x = x + 75
            click_y = y + 500
        else:
            raise ValueError(f"알 수 없는 클릭 태스크: {task}")

        win32api.SetCursorPos((click_x, click_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, click_x, click_y, 0, 0)
        time.sleep(self.mouse_delay)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, click_x, click_y, 0, 0)

    def _get_window_rect(self):
        """창의 클라이언트 영역 위치와 크기를 반환합니다."""
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd == 0:
            raise Exception(f"창을 찾을 수 없습니다: '{self.window_title}'")

        rect = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        return x, y, w, h

    def _action(self, action: int):
        """
        액션에 따라 키보드 입력을 처리합니다.

        Args:
            action: 0=왼쪽, 1=정지, 2=오른쪽
        """
        self._focus()

        if action == LEFT_ARROW:
            # 왼쪽 누름 + 오른쪽 뗌
            win32api.keybd_event(win32con.VK_LEFT, 0, 0, 0)
            win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)
        elif action == STOP:
            # 왼쪽 뗌 + 오른쪽 뗌
            win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)
        elif action == RIGHT_ARROW:
            # 왼쪽 뗌 + 오른쪽 누름
            win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_RIGHT, 0, 0, 0)
        else:
            raise ValueError(f"알 수 없는 액션: {action}")

    # ─────────────────────────────────────────────
    # gymnasium 인터페이스 구현
    # ─────────────────────────────────────────────

    def reset(self, seed=None, options=None, f5_reset=False):
        """
        에피소드를 초기화합니다.

        게임 화면을 클릭하고 스페이스바를 눌러 새 게임을 시작합니다.

        Args:
            f5_reset: True이면 F5로 페이지를 새로고침 후 시작 (게임 오류 시 사용)

        Returns:
            (obs, info): 초기 관측값과 빈 info 딕셔너리
        """
        self._focus()

        if f5_reset:
            # 페이지 새로고침 후 게임 로딩 대기
            self.perform("f5", delay=10.0)

        # 모든 키 입력 해제
        win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)

        # 게임 화면 클릭 → 스페이스바로 게임 시작
        self.perform("click_screen", delay=0.5)
        self.perform("press_space", delay=1.0)  # 게임 시작 대기
        self.perform("press_space", delay=0.2)  # 한 번 더 (안정성)

        # 초기 관측값
        obs = self.get_stacked_buffer()

        # 에피소드 추적 변수 초기화
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_distance = 0.0
        self.current_state = None

        # 타이밍 초기화
        self.next_step_time = time.perf_counter()
        self.last_step_time = time.perf_counter()
        self.step_frame_times.clear()
        self.last_min_h = 0

        return obs, {}

    def step(self, action):
        """
        액션을 실행하고 다음 관측값, 보상, 종료 여부를 반환합니다.

        보상 구조:
          - 생존 시: 0.5 - (min_h * 0.01)
              min_h: 화면 위에서 황새까지의 거리 (작을수록 황새가 위에 있음)
              황새가 높이 있을수록 (min_h가 작을수록) 보상이 큼
          - 사망 시: -500 + 진행 거리(m)
              오래 걸을수록 패널티가 줄어듦

        Args:
            action (int): 0=왼쪽, 1=정지, 2=오른쪽

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # FPS 기반 타이밍 제어 초기화
        if self.next_step_time is None:
            self.next_step_time = time.perf_counter()

        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        min_h = self.last_min_h  # 기본값: 이전 값 유지

        # 키보드 입력 적용
        self._action(action)

        # 현재 화면 프레임 가져오기
        obs_next = self.get_stacked_buffer()
        if obs_next is None:
            return obs_next, reward, terminated, truncated, info

        # 초기 상태 설정
        if self.current_state is None:
            self.current_state = LIVE

        # ─────────────────────────────────────────────
        # 사망 감지: 4개의 스택 프레임이 모두 동일하면 화면이 멈춘 것 → 사망
        # ─────────────────────────────────────────────
        if self._is_dead(obs_next):
            # OCR로 최종 진행 거리 읽기
            frame_org = self.capture.get_org_frame()
            final_distance = self._read_distance_ocr(frame_org[330:660, 100:1000])
            if final_distance is not None:
                self.episode_distance = final_distance

            self.current_state = DIE
            # 사망 보상: 기본 패널티 -500에 진행 거리만큼 부분 보상
            reward = -500.0 + self.episode_distance
            terminated = True
            print()  # 인라인 FPS 출력 후 줄바꿈
        else:
            # 생존 - 황새 위치 기반 보상 계산
            self.current_state = LIVE

            # 마지막 프레임에서 흰색 픽셀의 최상단 행 위치를 찾음
            # mask: 각 픽셀이 True(흰색)이면 황새가 있다는 의미
            mask = obs_next[..., -1] > 0
            row_has_white = mask.any(axis=1)  # 각 행에 흰색이 있는지

            if row_has_white.any():
                min_h = np.argmax(row_has_white)  # 첫 번째로 True인 행 (가장 위에 있는 황새)
                self.last_min_h = min_h
            else:
                # 황새를 찾지 못하면 이전 값 사용
                min_h = self.last_min_h

            # 생존 보상: 황새가 위에 있을수록 (min_h가 작을수록) 보상 증가
            reward = 0.5 - (min_h * 0.01)

        # 에피소드 누적 통계 업데이트
        self.episode_reward += reward
        self.episode_length += 1

        # 에피소드 종료 시 info 딕셔너리 구성
        if terminated or truncated:
            # SB3의 Monitor 래퍼와 호환되는 포맷
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }
            info["distance"] = self.episode_distance
            print(
                f"에피소드 종료 | 보상: {self.episode_reward:.2f} | "
                f"스텝: {self.episode_length} | 거리: {self.episode_distance:.2f}m"
            )
            time.sleep(2.0)  # 사망 후 잠시 대기 (다음 reset 전)

        # ─────────────────────────────────────────────
        # FPS 기반 타이밍 제어
        # 목표 FPS보다 빠르면 대기, 느리면 즉시 진행
        # ─────────────────────────────────────────────
        self.next_step_time += self.frame_interval
        sleep_time = self.next_step_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        # 실제 스텝 FPS 계산
        current_time = time.perf_counter()
        if self.last_step_time is not None:
            step_time = current_time - self.last_step_time
            self.step_frame_times.append(step_time)
            avg = sum(self.step_frame_times) / len(self.step_frame_times)
            self.current_step_fps = 1.0 / avg if avg > 0 else 0.0
        self.last_step_time = current_time

        # 실시간 FPS 및 보상 출력 (줄바꿈 없이)
        if not terminated:
            print(
                f"{self.get_fps_info()} | reward: {reward:.3f} | min_h: {min_h}",
                end="\r",
            )

        return obs_next, reward, terminated, truncated, info

    # ─────────────────────────────────────────────
    # 보조 메서드
    # ─────────────────────────────────────────────

    def _is_dead(self, obs_next):
        """
        사망 감지: 스택된 프레임들이 모두 동일하면 화면이 멈춘 것으로 판단합니다.

        게임 오버 시 화면이 정지되므로, 연속된 프레임이 변화 없이
        모두 동일하면 황새가 떨어진 것으로 봅니다.

        Args:
            obs_next: 스택된 관측값 (H, W, C * stacked_num)

        Returns:
            bool: 사망이면 True
        """
        if obs_next is None:
            return False

        num_channels = 1 if self.gray_scale else 3
        frames = []
        for i in range(self.stacked_num):
            start_ch = i * num_channels
            end_ch = (i + 1) * num_channels
            # 하단 영역(100: 이후)만 비교 (상단의 거리 표시 영역 제외)
            frame = obs_next[100:, :, start_ch:end_ch]
            frames.append(frame)

        # 모든 프레임이 첫 번째 프레임과 같으면 사망
        first_frame = frames[0]
        for frame in frames[1:]:
            if not np.array_equal(first_frame, frame):
                return False

        return True

    def _read_distance_ocr(self, distance_frame):
        """
        게임 화면의 거리 표시 영역에서 OCR로 진행 거리를 읽습니다.

        Args:
            distance_frame: 거리 표시 영역 이미지

        Returns:
            float: 거리(m), 읽기 실패 시 None
        """
        try:
            # 이진화로 숫자 인식률 향상
            _, thresh = cv2.threshold(distance_frame, 150, 255, cv2.THRESH_BINARY)

            results = self.ocr_reader.readtext(
                thresh,
                allowlist="0123456789.m",
                detail=0,
                paragraph=False,
            )

            if results:
                text = results[0].replace(" ", "").replace("O", "0")
                numbers = re.findall(r"\d+\.?\d*", text)
                if numbers:
                    distance = float(numbers[0])
                    if distance >= 0.0:
                        return distance
        except Exception as e:
            print(f"[OCR 오류] {e}")

        return None

    def get_fps_info(self):
        """스텝 FPS와 캡처 FPS 정보를 문자열로 반환합니다."""
        return f"Step FPS: {self.current_step_fps:.2f} (Obs: {self.capture.get_current_fps():.2f})"


# ─────────────────────────────────────────────
# 환경 단독 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    환경 단독 테스트: 랜덤 액션으로 에피소드를 실행합니다.
    사전에 Chrome에서 게임이 실행 중이어야 합니다.
    """
    print("=" * 60)
    print("WalkTheStorkEnv 환경 테스트 (랜덤 액션)")
    print("=" * 60)

    env = WalkTheStorkEnv(
        fps=30,
        crop_pos=(300, 600, 700, 450),
        resize=(320, 180),
        action_num=3,
        stacked_num=4,
        gray_scale=True,
    )

    print(f"액션 공간: {env.action_space}")
    print(f"관측 공간: {env.observation_space}")
    print("\n캡처 스레드 시작 중...")
    env.run_capture()

    episode = 0

    try:
        while True:
            episode += 1
            obs, info = env.reset()
            print(f"\n에피소드 {episode} 시작 | obs shape: {obs.shape if obs is not None else None}")

            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    print(f"에피소드 {episode} 종료 | 거리: {info.get('distance', 0):.2f}m")
                    break

                if obs is None:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n테스트 종료")
    finally:
        env.close()
