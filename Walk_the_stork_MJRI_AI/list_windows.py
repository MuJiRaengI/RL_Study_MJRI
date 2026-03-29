"""현재 열려있는 모든 윈도우 제목을 나열하는 유틸리티"""

import win32gui


def list_all_windows():
    """모든 가시적인 윈도우 제목 목록 반환"""

    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:  # 빈 제목 제외
                windows.append(title)
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    return sorted(set(windows))  # 중복 제거 및 정렬


if __name__ == "__main__":
    print("=" * 60)
    print("현재 열려있는 모든 윈도우 목록")
    print("=" * 60)

    windows = list_all_windows()

    for i, title in enumerate(windows, 1):
        print(f"{i:3d}. {title}")

    print(f"\n총 {len(windows)}개의 윈도우가 발견되었습니다.")
