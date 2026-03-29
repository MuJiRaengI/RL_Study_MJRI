import numpy as np
import mss
import win32gui
import win32con
import win32api
import time


class WindowCapture:
    def __init__(self, window_title: str):
        """게임 창 캡처 및 입력 시뮬레이션 환경 초기화

        Args:
            window_title (str): 캡처할 창 제목
        """
        self.window_title = window_title

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

        # 창을 foreground로 가져오기 (에러 무시)
        try:
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception as e:
            print(f"SetForegroundWindow 경고 (무시됨): {e}")
            # 대안: Alt 키를 눌렀다 떼는 것으로 제한 우회
            try:
                win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.05)
                win32gui.SetForegroundWindow(self.hwnd)
            except:
                print("창 포커스 설정 실패 - 수동으로 창을 클릭해주세요")

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
        return img

    def get_frame(self):
        """원본 프레임 반환

        Returns:
            numpy.ndarray: 원본 프레임 (없으면 None)
        """
        return self._capture_window()
