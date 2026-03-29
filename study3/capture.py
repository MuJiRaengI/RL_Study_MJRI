"""
study3/capture.py - 게임 화면 캡처 모듈
=========================================
Chrome 브라우저에서 실행 중인 게임 창을 실시간으로 캡처하여
RL 에이전트의 관측값(observation)으로 제공합니다.

핵심 클래스:
  GameCapture: 별도 스레드에서 화면을 지정 FPS로 캡처하고,
               최근 N개 프레임을 버퍼에 저장합니다.
               get_stacked_frames()로 프레임들을 채널 방향으로
               스택한 배열을 반환합니다.
"""

import numpy as np
import cv2
import mss
import win32gui
import win32con
import time
import threading
from collections import deque


class GameCapture(threading.Thread):
    """
    게임 창을 백그라운드 스레드에서 지속적으로 캡처하는 클래스.

    daemon 스레드로 실행되므로 메인 프로세스가 종료되면 자동으로 종료됩니다.
    start() 대신 GameCapture를 생성한 후 run_capture()를 통해 시작하세요.
    (환경에서 run_capture()를 호출합니다)
    """

    def __init__(
        self,
        window_title: str,
        fps: int = 30,
        buffer_length: int = 4,
        crop_pos: tuple = None,
        resize: tuple = None,
        gray_scale: bool = False,
        skip_duplicate_frames: bool = False,
    ):
        """
        Args:
            window_title (str): 캡처할 창 제목 (정확히 일치해야 함)
            fps (int): 캡처 FPS (목표치, 실제는 시스템 성능에 따라 다를 수 있음)
            buffer_length (int): 보관할 최근 프레임 수 (프레임 스택 크기)
            crop_pos (tuple): 크롭 영역 (x, y, width, height), None이면 크롭 안함
            resize (tuple): 리사이즈 크기 (width, height), None이면 리사이즈 안함
            gray_scale (bool): True이면 그레이스케일로 변환
            skip_duplicate_frames (bool): True이면 이전과 동일한 프레임 제외
        """
        super().__init__(daemon=True)
        self.window_title = window_title
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.buffer_length = buffer_length
        self.crop_pos = crop_pos
        self.resize = resize
        self.gray_scale = gray_scale
        self.skip_duplicate_frames = skip_duplicate_frames

        # 원본 프레임 (OCR 등 전처리 없이 사용할 때)
        self.frame_org = None

        # 프레임 버퍼: 최근 buffer_length개 프레임을 FIFO로 보관
        self.frame_buffer = deque(maxlen=buffer_length)
        self.lock = threading.Lock()
        self.org_frame_lock = threading.Lock()

        # 스레드 제어
        self.running = False

        # FPS 측정용
        self.frame_times = deque(maxlen=30)  # 최근 30프레임 이동 평균
        self.current_fps = 0.0
        self.last_frame_time = None

        # 창 정보 캐싱 (매번 FindWindow 호출 방지)
        self.hwnd = None
        self.monitor_region = None
        self.window_height = None
        self.window_width = None

        # mss 인스턴스 (run() 스레드 내에서 생성)
        self.sct = None

        # 초기화 (창 찾기)
        self._initialize_window()

    def _initialize_window(self):
        """창 핸들과 영역 정보 초기화 (한 번만 실행)"""
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        if self.hwnd == 0:
            raise Exception(
                f"창을 찾을 수 없습니다: '{self.window_title}'\n"
                f"Chrome에서 게임이 실행 중인지 확인하세요."
            )

        # 클라이언트 영역 크기 (타이틀바 제외)
        rect = win32gui.GetClientRect(self.hwnd)
        x, y = win32gui.ClientToScreen(self.hwnd, (rect[0], rect[1]))
        w, h = rect[2] - rect[0], rect[3] - rect[1]

        # mss가 사용할 캡처 영역
        self.monitor_region = {"top": y, "left": x, "width": w, "height": h}
        self.window_height = h
        self.window_width = w

    def _capture_window(self):
        """창 화면을 캡처하고 전처리(크롭, 리사이즈, 그레이스케일)를 수행합니다."""
        # mss 인스턴스는 스레드 내에서 생성 (스레드 안전)
        if self.sct is None:
            self.sct = mss.mss()

        screenshot = self.sct.grab(self.monitor_region)

        # (H, W, 3) RGB 배열로 변환
        img = np.frombuffer(screenshot.rgb, dtype=np.uint8)
        img = img.reshape((self.window_height, self.window_width, 3))

        # 원본 프레임 저장 (OCR 등에서 사용)
        with self.org_frame_lock:
            self.frame_org = img

        # 크롭: (x, y, width, height) → img[y:y+h, x:x+w]
        if self.crop_pos is not None:
            x, y, w, h = self.crop_pos
            img = img[y : y + h, x : x + w]

        # 그레이스케일 변환
        if self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 리사이즈
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_NEAREST)

        # 그레이스케일이면 채널 차원 추가: (H, W) → (H, W, 1)
        if self.gray_scale:
            img = np.expand_dims(img, axis=-1)

        return img

    def get_org_frame(self):
        """전처리 이전 원본 프레임을 반환합니다 (OCR 등에 사용)."""
        with self.org_frame_lock:
            return self.frame_org

    def run(self):
        """
        스레드 실행 함수: 목표 FPS로 화면을 지속적으로 캡처합니다.

        누적 오차 보정(cumulative error correction) 방식으로
        정확한 FPS를 유지합니다.
        """
        self.running = True
        print(f"GameCapture 스레드 시작 (FPS: {self.fps}, 버퍼: {self.buffer_length})")

        frame_interval = 1.0 / self.fps
        next_frame_time = time.perf_counter()  # 다음 프레임의 목표 시간
        self.last_frame_time = time.perf_counter()

        while self.running:
            try:
                next_frame_time += frame_interval  # 목표 시간 누적 (드리프트 방지)

                frame = self._capture_window()

                # 중복 프레임 제외 옵션
                should_append = True
                if self.skip_duplicate_frames and len(self.frame_buffer) > 0:
                    with self.lock:
                        last_frame = self.frame_buffer[-1]
                    if np.array_equal(frame, last_frame):
                        should_append = False

                if should_append:
                    with self.lock:
                        self.frame_buffer.append(frame)

                # 다음 목표 시간까지 대기
                sleep_time = next_frame_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # sleep_time < 0 이면 (처리 지연) → 즉시 다음 프레임 처리

                # FPS 계산 (실제 캡처 속도)
                current_time = time.perf_counter()
                frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time

                self.frame_times.append(frame_time)
                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

            except Exception as e:
                print(f"캡처 오류: {e}")
                time.sleep(0.1)
                next_frame_time = time.perf_counter()  # 오류 후 타이밍 재설정

        print("GameCapture 스레드 종료")

    def stop(self):
        """캡처 스레드를 중지하고 리소스를 정리합니다."""
        self.running = False
        self.join(timeout=1.0)

        # mss 인스턴스 정리 (스레드 로컬 객체이므로 try-except)
        if self.sct is not None:
            try:
                self.sct.close()
            except (AttributeError, RuntimeError):
                pass
            self.sct = None

    def get_screen(self):
        """
        가장 최근에 캡처된 프레임 1개를 반환합니다.

        Returns:
            numpy.ndarray: 최근 프레임 (버퍼가 비어 있으면 None)
        """
        if self.running:
            with self.lock:
                if len(self.frame_buffer) > 0:
                    return self.frame_buffer[-1]
                return None
        else:
            # 스레드가 실행 중이 아닐 때는 직접 캡처
            return self._capture_window()

    def get_buffer(self):
        """
        버퍼에 저장된 모든 프레임을 리스트로 반환합니다 (오래된 순서).

        Returns:
            list: 프레임 리스트
        """
        with self.lock:
            return list(self.frame_buffer)

    def get_stacked_frames(self):
        """
        버퍼의 프레임들을 채널 방향으로 스택(concatenate)하여 반환합니다.

        예) buffer_length=4, gray_scale=True, resize=(320,180):
            각 프레임: (180, 320, 1)
            스택 결과: (180, 320, 4)  ← 4프레임을 채널로 합침

        Returns:
            numpy.ndarray: (H, W, C * buffer_length) 형태의 배열
                          버퍼가 가득 차지 않았으면 None
        """
        with self.lock:
            if len(self.frame_buffer) == self.buffer_length:
                frames = list(self.frame_buffer)
            else:
                return None

        # lock 외부에서 concat (오래 걸리는 작업은 lock 밖에서)
        return np.concatenate(frames, axis=-1)

    def get_current_fps(self):
        """현재 실시간 캡처 FPS를 반환합니다."""
        return self.current_fps
