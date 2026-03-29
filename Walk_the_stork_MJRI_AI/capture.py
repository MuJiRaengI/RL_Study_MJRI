import numpy as np
import cv2
import mss
import win32gui
import win32con
import win32api
import time
import threading
from collections import deque


class GameFPSMeasurer:
    """실제 게임 FPS 측정기 (프레임 변화 감지 기반)"""

    def __init__(self, window_size=30):
        """
        Args:
            window_size: FPS 계산에 사용할 프레임 변화 횟수 (이동 평균)
        """
        self.window_size = window_size
        self.last_frame = None
        self.last_change_time = None
        self.change_intervals = deque(maxlen=window_size)
        self.game_fps = 0.0

    def update(self, frame):
        """
        새 프레임을 받아 게임 FPS 업데이트

        Args:
            frame: numpy array (게임 프레임)

        Returns:
            bool: 프레임이 변경되었으면 True, 동일하면 False
        """
        current_time = time.perf_counter()

        if self.last_frame is not None:
            # 프레임이 변경되었는지 확인
            if not np.array_equal(frame, self.last_frame):
                # 프레임 변경 감지
                if self.last_change_time is not None:
                    interval = current_time - self.last_change_time
                    self.change_intervals.append(interval)

                    # FPS 계산
                    if len(self.change_intervals) > 0:
                        avg_interval = sum(self.change_intervals) / len(
                            self.change_intervals
                        )
                        self.game_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0

                self.last_change_time = current_time
                self.last_frame = frame.copy()
                return True
            else:
                # 프레임 동일
                return False
        else:
            # 첫 프레임
            self.last_frame = frame.copy()
            self.last_change_time = current_time
            return True

    def get_fps(self):
        """현재 측정된 게임 FPS 반환"""
        return self.game_fps

    def reset(self):
        """측정기 초기화"""
        self.last_frame = None
        self.last_change_time = None
        self.change_intervals.clear()
        self.game_fps = 0.0


class GameCapture(threading.Thread):
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
        """게임 창 캡처 및 입력 시뮬레이션 환경 초기화

        Args:
            window_title (str): 캡처할 창 제목
            fps (int): 캡처 FPS
            buffer_length (int): 저장할 프레임 개수 (deque 크기)
            crop_pos (tuple): 크롭 위치 (x, y, width, height), None이면 크롭 안함
            resize (tuple): 리사이즈 크기 (width, height), None이면 리사이즈 안함
            gray_scale (bool): True인 경우 그레이스케일로 변환
            skip_duplicate_frames (bool): True인 경우 이전 프레임과 동일한 프레임은 버퍼에 추가하지 않음
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
        self.frame_org = None

        # 프레임 버퍼 (최근 N개 프레임 저장)
        self.frame_buffer = deque(maxlen=buffer_length)
        self.lock = threading.Lock()
        self.org_frame_lock = threading.Lock()

        # 스레드 제어
        self.running = False
        self.last_capture_time = 0

        # FPS 측정용
        self.frame_times = deque(maxlen=30)  # 최근 30프레임 평균
        self.current_fps = 0.0
        self.last_frame_time = None

        # 윈도우 정보 캐싱
        self.hwnd = None
        self.monitor_region = None
        self.window_height = None
        self.window_width = None

        # mss 인스턴스 (run() 스레드에서 생성)
        self.sct = None

        # 초기화
        self._initialize_window()

    def _initialize_window(self):
        """윈도우 핸들과 영역 정보 초기화 (한 번만 실행)"""
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        if self.hwnd == 0:
            raise Exception(f"Window '{self.window_title}' not found")

        # 창 영역 얻기 (클라이언트 영역)
        rect = win32gui.GetClientRect(self.hwnd)
        x, y = win32gui.ClientToScreen(self.hwnd, (rect[0], rect[1]))
        w, h = rect[2] - rect[0], rect[3] - rect[1]

        # 모니터 영역 저장
        self.monitor_region = {"top": y, "left": x, "width": w, "height": h}
        self.window_height = h
        self.window_width = w

    def _get_window_rect(self):
        """창 핸들과 위치/크기 얻기"""
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd == 0:
            raise Exception(f"Window '{self.window_title}' not found")

        # 창을 foreground로 가져오기 (에러 무시)
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:
            # 에러 발생해도 계속 진행
            pass

        # 창 영역 얻기 (클라이언트 영역)
        rect = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
        w, h = rect[2] - rect[0], rect[3] - rect[1]

        return x, y, w, h

    def _capture_window(self):
        """특정 창 캡처 (최적화됨)"""
        # sct가 없으면 생성 (스레드 로컬)
        if self.sct is None:
            self.sct = mss.mss()

        screenshot = self.sct.grab(self.monitor_region)

        # NumPy 배열로 변환
        img = np.frombuffer(screenshot.rgb, dtype=np.uint8)
        img = img.reshape((self.window_height, self.window_width, 3))

        with self.org_frame_lock:
            self.frame_org = img

        if self.crop_pos is not None:
            x, y, w, h = self.crop_pos
            img = img[y : y + h, x : x + w]

        # Gray 스케일 변환
        if self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 리사이즈 (필요시)
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_NEAREST)

        if self.gray_scale:
            # (H, W) -> (H, W, 1)
            img = np.expand_dims(img, axis=-1)

        return img

    def get_org_frame(self):
        """원본 프레임 반환

        Returns:
            numpy.ndarray: 원본 프레임 (없으면 None)
        """
        with self.org_frame_lock:
            return self.frame_org

    def run(self):
        """스레드 실행 시 지속적으로 화면 캡처"""
        self.running = True
        print(
            f"GameCapture 스레드 시작 (FPS: {self.fps}, Buffer: {self.buffer_length})"
        )

        # 누적 오차 보정 방식으로 정확한 FPS 유지
        frame_interval = 1.0 / self.fps
        next_frame_time = time.perf_counter()  # 다음 프레임 목표 시간
        self.last_frame_time = time.perf_counter()

        while self.running:
            try:
                next_frame_time += frame_interval  # 목표 시간 누적

                frame = self._capture_window()

                # 마지막 프레임과 비교 (lock 최소화)
                should_append = True
                if self.skip_duplicate_frames and len(self.frame_buffer) > 0:
                    last_frame = self.frame_buffer[-1]
                    # lock 내부에서 비교 수행
                    if np.array_equal(frame, last_frame):
                        should_append = False

                # 다른 프레임이면 추가
                if should_append:
                    with self.lock:
                        self.frame_buffer.append(frame)

                # 누적 오차 보정: 다음 목표 시간까지 대기
                current_time = time.perf_counter()
                sleep_time = next_frame_time - current_time

                if sleep_time > 0:
                    time.sleep(sleep_time)
                # sleep_time이 음수면 (목표보다 느림) → sleep 없이 최대 속도로 실행

                # FPS 계산
                current_time = time.perf_counter()
                frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time

                self.frame_times.append(frame_time)
                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    self.current_fps = (
                        1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    )

            except Exception as e:
                print(f"캡처 오류: {e}")
                time.sleep(0.1)
                # 오류 발생 시 타이밍 재설정
                next_frame_time = time.perf_counter()

        print("GameCapture 스레드 종료")

    def stop(self):
        """스레드 중지"""
        self.running = False
        self.join(timeout=1.0)

        # mss 인스턴스 정리 (스레드 로컬 객체이므로 try-except로 안전하게 처리)
        if self.sct is not None:
            try:
                self.sct.close()
            except (AttributeError, RuntimeError):
                # 스레드 로컬 객체 접근 오류는 무시 (이미 스레드가 종료됨)
                pass
            self.sct = None

    def get_screen(self):
        """가장 최근 캡처된 프레임 반환

        Returns:
            numpy.ndarray: 최근 프레임 (없으면 None)
        """
        # 스레드가 실행 중이면 버퍼에서 가져오기
        if self.running:
            with self.lock:
                if len(self.frame_buffer) > 0:
                    return self.frame_buffer[-1]
                return None
        # 스레드가 실행 중이 아니면 직접 캡처
        else:
            return self._capture_window()

    def get_buffer(self):
        """버퍼에 저장된 모든 프레임 반환

        Returns:
            list: 프레임 리스트 (오래된 순서대로)
        """
        with self.lock:
            return list(self.frame_buffer)

    def get_stacked_frames(self):
        """버퍼의 프레임들을 채널 방향으로 스택

        Returns:
            numpy.ndarray: (H, W, C*buffer_length) 형태의 배열
                          버퍼가 가득 차지 않았으면 None
        """
        with self.lock:
            if len(self.frame_buffer) == self.buffer_length:
                frames = list(self.frame_buffer)
            else:
                return None

        # lock 외부에서 concat 수행 (시간이 오래 걸리는 작업)
        return np.concatenate(frames, axis=-1)

    def get_current_fps(self):
        """현재 실시간 FPS 반환"""
        return self.current_fps


def test():
    """GameCapture 테스트 함수"""
    print("=" * 60)
    print("GameCapture FPS 테스트")
    print("=" * 60)
    window_title = (
        "황새 오래 걷기 (Walk the Stork) - 플래시게임 | 와플래시 게임 아카이브 - Chrome"
    )

    print(f"\n🔍 '{window_title}' 창을 찾는 중...")

    try:
        # GameCapture 인스턴스 생성 (30 FPS, 버퍼 4, 320x180 리사이즈, 그레이스케일)
        capture = GameCapture(
            window_title=window_title,
            fps=30,
            buffer_length=4,
            crop_pos=(100, 300, 1050, 750),
            resize=(320, 180),
            gray_scale=True,
        )

        print(f"✅ 창 발견! ({capture.window_width}x{capture.window_height})")
        print(f"📷 캡처 설정: 60 FPS, 버퍼 크기 4, 리사이즈 320x180, 그레이스케일")
        print("\n🚀 캡처 시작 (Ctrl+C로 종료)...")
        print("-" * 60)

        # 스레드 시작
        capture.start()

        # 버퍼가 가득 찰 때까지 대기
        while len(capture.frame_buffer) < capture.buffer_length:
            time.sleep(0.1)

        print("✅ 버퍼 준비 완료!\n")

        # 주기적으로 상태 출력
        while True:
            buffer_size = len(capture.frame_buffer)
            current_fps = capture.get_current_fps()

            # 최신 프레임 정보
            latest_frame = capture.get_screen()
            if latest_frame is not None:
                shape = latest_frame.shape
                print(
                    f"📊 버퍼: {buffer_size}/{capture.buffer_length} | "
                    f"FPS: {current_fps:6.2f} | "
                    f"프레임 크기: {shape}",
                    end="\r",
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n⏹️  중지 요청 받음. 종료 중...")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        if "capture" in locals():
            capture.stop()
        print("✅ 종료 완료.")


def test_game_fps():
    """실제 게임 FPS 측정 테스트"""
    print("=" * 60)
    print("게임 실제 FPS 측정")
    print("=" * 60)
    window_title = (
        "황새 오래 걷기 (Walk the Stork) - 플래시게임 | 와플래시 게임 아카이브 - Chrome"
    )

    print(f"\n🔍 '{window_title}' 창을 찾는 중...")

    try:
        # GameCapture 인스턴스 생성 (원본 프레임 캡처용)
        capture = GameCapture(
            window_title=window_title,
            fps=60,  # 높은 FPS로 캡처해서 게임 변화를 놓치지 않음
            buffer_length=4,
            crop_pos=(100, 300, 1050, 750),
            resize=(320, 180),
            gray_scale=True,
        )

        # GameFPSMeasurer 인스턴스 생성
        fps_measurer = GameFPSMeasurer(window_size=30)

        print(f"✅ 창 발견! ({capture.window_width}x{capture.window_height})")
        print(f"📷 캡처 설정: 60 FPS (높은 캡처율로 게임 프레임 변화 감지)")
        print("\n🚀 게임 FPS 측정 시작 (Ctrl+C로 종료)...")
        print("-" * 60)

        # 스레드 시작
        capture.start()

        # 버퍼가 준비될 때까지 대기
        time.sleep(1.0)

        print("✅ 측정 준비 완료!\n")

        frame_count = 0
        last_print_time = time.perf_counter()

        # 주기적으로 프레임 확인하며 게임 FPS 측정
        while True:
            # 최신 프레임 가져오기
            frame = capture.get_screen()

            if frame is not None:
                # FPS 측정기에 프레임 전달
                changed = fps_measurer.update(frame)
                frame_count += 1

                # 1초마다 출력
                current_time = time.perf_counter()
                if current_time - last_print_time >= 1.0:
                    capture_fps = capture.get_current_fps()
                    game_fps = fps_measurer.get_fps()

                    print(
                        f"🎮 실제 게임 FPS: {game_fps:6.2f} | "
                        f"캡처 FPS: {capture_fps:6.2f} | "
                        f"측정 프레임: {frame_count:5d}",
                        end="\r",
                    )
                    last_print_time = current_time

            time.sleep(1.0 / 120)  # 60Hz로 체크

    except KeyboardInterrupt:
        print("\n\n⏹️  중지 요청 받음. 종료 중...")
        if "fps_measurer" in locals():
            print(f"\n📊 최종 측정 결과:")
            print(f"   실제 게임 FPS: {fps_measurer.get_fps():.2f}")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "capture" in locals():
            capture.stop()
        print("✅ 종료 완료.")


if __name__ == "__main__":
    # test()  # 캡처 테스트
    test_game_fps()  # 게임 FPS 측정
