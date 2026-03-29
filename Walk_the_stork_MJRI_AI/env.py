import gymnasium as gym
from gymnasium import spaces
import numpy as np
import win32gui
import win32con
import win32api
import time
import cv2
import os
import threading
import torch
from collections import deque
import easyocr
import re

from capture import GameCapture
from consts import *


class WalkTheStork(gym.Env):
    def __init__(
        self,
        fps=30,
        crop_pos=(100, 300, 1050, 750),
        resize=(320, 180),
        action_num=3,
        stacked_num=4,
        gray_scale: bool = True,
        device="cuda:0",
    ):
        super().__init__()
        self.window_title = "황새 오래 걷기 (Walk the Stork) - 플래시게임 | 와플래시 게임 아카이브 - Chrome"
        self.crop_pos = crop_pos
        self.resize = resize
        self.fps = fps
        self.stacked_num = stacked_num
        self.gray_scale = gray_scale
        self.device = device if torch.cuda.is_available() else "cpu"
        # 0: left arrow, 1: stop 2: right arrow
        self.action_space = spaces.Discrete(action_num)

        # Observation: 마지막 (x, y) + y 변화량 (N-1)개
        ch = 3
        if self.gray_scale:
            ch = 1
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.resize[1], self.resize[0], ch * self.stacked_num),
            dtype=np.uint8,
        )

        # EasyOCR 초기화 (죽음 시 최종 거리 기록용)
        print("EasyOCR 초기화 중...")
        self.ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("EasyOCR 초기화 완료!")

        self.key_delay = 0.05
        self.mouse_delay = 0.05
        self.capture = GameCapture(
            self.window_title,
            self.fps,
            self.stacked_num,
            self.crop_pos,
            self.resize,
            self.gray_scale,
        )

        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_distance = 0.0

        self.current_state = None

        # Step 타이밍 제어용
        self.frame_interval = 1.0 / self.fps
        self.next_step_time = None

        # Step FPS 측정용 (더 긴 윈도우로 안정화)
        self.step_frame_times = deque(maxlen=100)  # 100 프레임 평균
        self.current_step_fps = 0.0
        self.last_step_time = None

        self.first_x_threshold = 130
        self.last_x_threshold = 210
        self.last_min_h = 0  # 마지막으로 계산된 min_h 값

    def run_capture(self):
        """캡처 스레드 시작"""
        self.capture.start()
        time.sleep(1.0)  # 안정화 딜레이

    def get_stacked_buffer(self):
        frames = self.capture.get_stacked_frames()

        if frames is None:
            return None

        # 검은색만 추출 (threshold 적용)
        threshold_value = 30
        frames_black = (frames < threshold_value).astype(np.uint8) * 255

        return frames_black

    def perform(self, task: str, delay: float = None):
        """태스크에 따라 적절한 액션 수행"""
        if task in ["press_space", "f5"]:
            if delay is None:
                delay = self.key_delay
            self._key(task, delay)
        elif task in ["click_screen"]:
            self._click(task)
        elif task in ["pause"]:
            # print("일시정지")
            pass
        elif task in ["continue"]:
            # print("재개")
            pass
        else:
            raise ValueError(f"Unknown task: {task}")

    def _focus(self):
        """창 포커스 맞추기"""
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.01)  # 포커스 안정화 딜레이
        except Exception as e:
            pass

    def _key(self, task: str, delay: float = None):
        """키보드 입력 시뮬레이션"""
        # 창 포커스 맞추기
        self._focus()

        if task in ["press_space"]:
            # 스페이스바 누름
            # print("스페이스바 누름")
            win32api.keybd_event(win32con.VK_SPACE, 0, 0, 0)
            time.sleep(self.key_delay)
            win32api.keybd_event(win32con.VK_SPACE, 0, win32con.KEYEVENTF_KEYUP, 0)

        elif task in ["f5"]:
            # F5 누르름
            print("F5 누르름")
            win32api.keybd_event(win32con.VK_F5, 0, 0, 0)
            time.sleep(self.key_delay)
            win32api.keybd_event(win32con.VK_F5, 0, win32con.KEYEVENTF_KEYUP, 0)

        else:
            raise ValueError(f"Unknown task: {task}")

        if delay is not None and delay > 0:
            time.sleep(delay)

    def _click(self, task: str):
        """창 내 특정 좌표 클릭"""
        x, y, w, h = self._get_window_rect()
        if task == "click_screen":
            # print("click_screen")
            click_x = x + 75
            click_y = y + 500
            delay = None
        else:
            raise ValueError(f"Unknown task: {task}")

        # 커서 이동
        win32api.SetCursorPos((click_x, click_y))

        # 왼쪽 클릭 (다운 -> 업)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, click_x, click_y, 0, 0)
        time.sleep(self.mouse_delay)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, click_x, click_y, 0, 0)
        if delay is not None and delay > 0:
            time.sleep(delay)

    def _get_window_rect(self):
        """창 핸들과 위치/크기 얻기"""
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd == 0:
            raise Exception(f"Window '{self.window_title}' not found")

        # 창 영역 얻기 (클라이언트 영역) - ShowWindow 호출 제거
        rect = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
        w, h = rect[2] - rect[0], rect[3] - rect[1]

        return x, y, w, h

    def _action(self, action: int):
        """액션에 따라 키 입력 처리

        Args:
            action: 0=왼쪽, 1=정지, 2=오른쪽
        """
        self._focus()

        if action == LEFT_ARROW:
            # 왼쪽 화살표 누르기, 오른쪽 화살표 떼기
            win32api.keybd_event(win32con.VK_LEFT, 0, 0, 0)
            win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)
        elif action == STOP:
            # 왼쪽, 오른쪽 모두 떼기
            win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)
        elif action == RIGHT_ARROW:
            # 왼쪽 화살표 떼기, 오른쪽 화살표 누르기
            win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_RIGHT, 0, 0, 0)
        else:
            raise ValueError(f"Unknown action: {action}")

    def reset(self, seed=None, f5_reset=False):
        # print("🔄 환경 리셋 중...")
        self._focus()
        if f5_reset:
            self.perform("f5", delay=10.0)

        # 왼쪽, 오른쪽 모두 떼기
        win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)
        self.perform("click_screen", delay=0.5)
        self.perform("press_space", delay=1.0)
        self.perform("press_space", delay=0.2)

        obs = self.get_stacked_buffer()

        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_distance = 0.0

        self.current_state = None

        # Step 타이밍 초기화
        self.next_step_time = time.perf_counter()
        self.last_step_time = time.perf_counter()
        self.step_frame_times.clear()
        self.last_min_h = 0  # min_h 초기화

        # print("✅ 환경 리셋 완료!")
        return obs, {}

    def _is_dead(self, obs_next):
        """
        죽음 감지: 스택에 쌓인 4개의 프레임이 모두 동일하면 죽음으로 판단
        (화면이 멈춘 상태 = 죽음)

        Args:
            obs_next: 스택된 관찰 (height, width, channels * stacked_num)

        Returns:
            bool: 죽었으면 True, 살았으면 False
        """
        if obs_next is None:
            return False

        # 각 프레임 분리 (stacked_num개)
        num_channels = 1 if self.gray_scale else 3
        frames = []
        for i in range(self.stacked_num):
            start_ch = i * num_channels
            end_ch = (i + 1) * num_channels
            frame = obs_next[100:, :, start_ch:end_ch]
            frames.append(frame)

        # 모든 프레임이 첫 번째 프레임과 동일한지 확인
        first_frame = frames[0]
        for frame in frames[1:]:
            if not np.array_equal(first_frame, frame):
                return False

        # 모든 프레임이 동일하면 죽음
        return True

    def _read_distance_ocr(self, distance_frame):
        """거리 표시 영역에서 OCR로 거리 읽기

        Args:
            distance_frame: 거리 표시 영역 프레임

        Returns:
            float: 읽은 거리 (m), 실패 시 None
        """
        try:
            # 이진화
            _, thresh = cv2.threshold(distance_frame, 150, 255, cv2.THRESH_BINARY)

            # EasyOCR 수행
            results = self.ocr_reader.readtext(
                thresh,
                allowlist="0123456789.m",
                detail=0,
                paragraph=False,
            )

            # 결과에서 숫자 추출
            if results:
                text = results[0].replace(" ", "").replace("O", "0")
                numbers = re.findall(r"\d+\.?\d*", text)
                if numbers:
                    distance = float(numbers[0])
                    if distance >= 0.0:
                        return distance
        except Exception as e:
            print(f"[OCR] 오류: {e}")

        return None

    def step(self, action):
        # print("🚶‍♂️ Step 액션 수행 중...")
        # 🕒 FPS 기반 타이밍 제어
        if self.next_step_time is None:
            self.next_step_time = time.perf_counter()

        reward = 0.0

        distance = 0.0
        good_position_reward = 0.5
        distance_penalty = 0.01
        terminated = False
        truncated = False
        info = {}
        options = {}

        self._action(action)

        obs_next = self.get_stacked_buffer()
        if obs_next is None:
            # print("⚠️ 관찰 프레임이 None입니다. 대기 후 재시도합니다.")
            return obs_next, reward, terminated, truncated, info

        # 초기 상태 설정
        if self.current_state is None:
            self.current_state = LIVE

        # 죽음 감지
        if self._is_dead(obs_next):
            # 죽음 - 최종 거리 OCR로 읽기
            frame_org = self.capture.get_org_frame()
            final_distance = self._read_distance_ocr(frame_org[330:660, 100:1000])
            if final_distance is not None:
                self.episode_distance = final_distance
                # print(f"⚰️ 최종 거리: {final_distance}m")

            self.current_state = DIE
            reward = -500.0 + self.episode_distance  # 죽음 패널티 + 진행 거리 보상
            terminated = True
            print()  # FPS 출력 후 개행
        else:
            # 생존
            self.current_state = LIVE

            mask = obs_next[..., -1] > 0

            row_has_white = mask.any(axis=1)  # 각 행에 흰색이 있는지 확인
            if row_has_white.any():
                min_h = np.argmax(row_has_white)  # 처음으로 True인 행
                self.last_min_h = min_h  # 계산 성공 시 저장
            else:
                min_h = self.last_min_h  # 계산 실패 시 이전 값 사용

            reward = good_position_reward - (min_h * distance_penalty)

        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }
            # Monitor wrapper가 덮어쓰므로 별도 키로도 저장
            info["distance"] = self.episode_distance
            print(
                f"🏁 에피소드 종료! 보상: {self.episode_reward:.2f}, 길이: {self.episode_length}, 진행거리: {self.episode_distance:.2f} m"
            )
            time.sleep(2.0)  # 죽음 후 잠시 대기

        # 🕒 다음 프레임 목표 시간까지 대기
        self.next_step_time += self.frame_interval
        sleep_time = self.next_step_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        # sleep_time이 음수면 (목표보다 느림) → sleep 없이 최대 속도로 실행

        # Step FPS 계산 (sleep 후, 전체 프레임 시간 측정)
        current_time = time.perf_counter()
        if self.last_step_time is not None:
            step_frame_time = current_time - self.last_step_time
            self.step_frame_times.append(step_frame_time)
            avg_step_time = sum(self.step_frame_times) / len(self.step_frame_times)
            self.current_step_fps = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        self.last_step_time = current_time

        # FPS 정보 출력 (개행 없이)
        if not terminated:
            print(
                f"{self.get_fps_info()} | reward : {reward:.3f} | min_h : {min_h}",
                end="\r",
            )

        # print("✅ Step 액션 수행 완료!")
        return obs_next, reward, terminated, truncated, info

    def get_fps_info(self):
        """스텝 FPS와 obs generator FPS 함께 반환"""
        step_fps = self.current_step_fps
        obs_fps = self.capture.get_current_fps()
        return f"Step FPS: {step_fps:.2f} (Obs: {obs_fps:.2f})"

    def print_fps_info(self):
        """스텝 FPS 정보 출력"""
        print(self.get_fps_info())


def test():
    """WalkTheStork 환경 테스트 함수"""
    print("=" * 60)
    print("WalkTheStork 환경 FPS 테스트")
    print("=" * 60)

    try:
        # 환경 생성
        env = WalkTheStork(
            fps=30,
            crop_pos=(100, 300, 1050, 750),
            resize=(320, 180),
            action_num=3,
            stacked_num=4,
            gray_scale=True,
            device="cuda:0",
        )

        print("\n✅ 환경 생성 완료!")
        print(f"📊 설정: FPS={env.fps}, 리사이즈={env.resize}, 스택={env.stacked_num}")
        print(f"🎮 액션 공간: {env.action_space}")
        print(f"👀 관찰 공간: {env.observation_space}")

        # 캡처 스레드 시작
        print("\n🚀 캡처 스레드 시작 중...")
        env.run_capture()

        # 환경 리셋
        print("🔄 환경 리셋 중...")
        obs, info = env.reset(f5_reset=False)

        print(f"✅ 초기 관찰: shape={obs.shape if obs is not None else None}")
        print("\n▶️  테스트 시작 (Ctrl+C로 종료)...")
        print("-" * 60)

        # 랜덤 액션으로 테스트
        step_count = 0
        action_names = {0: "LEFT", 1: "STOP", 2: "RIGHT"}

        while True:
            # 랜덤 액션 선택
            action = env.action_space.sample()

            # Step 실행
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # FPS 및 상태 정보 출력
            if step_count % 30 == 0:  # 30 step마다 출력
                print(
                    f"Step: {step_count:4d} | "
                    f"Action: {action_names[action]:5s} | "
                    f"Reward: {reward:7.3f} | "
                    f"Distance: {env.episode_distance:6.2f}m | "
                    f"{env.get_fps_info()}",
                    end="\r",
                )

            # 에피소드 종료 시
            if terminated or truncated:
                print(f"\n{'='*60}")
                print(f"Episode 종료 정보:")
                if "episode" in info:
                    print(f"  - 총 보상: {info['episode']['r']:.2f}")
                    print(f"  - 길이: {info['episode']['l']}")
                if "distance" in info:
                    print(f"  - 진행 거리: {info['distance']:.2f}")
                print(f"{'='*60}\n")

                # 리셋
                obs, info = env.reset()
                step_count = 0

            # 관찰이 None이면 대기
            if obs is None:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n⏹️  중지 요청 받음. 종료 중...")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "env" in locals():
            env.capture.stop()
        print("✅ 종료 완료.")


if __name__ == "__main__":
    test()
