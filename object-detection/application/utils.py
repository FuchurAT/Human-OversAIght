"""
Utility functions for video playback and multi-monitor support.
"""

import cv2
import numpy as np
import multiprocessing
from typing import List, Optional

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None


def _play_video_fullscreen_on_monitor(video_path: str, monitor_idx: int) -> None:
    """
    Play a video in fullscreen on the specified monitor index.
    """
    monitors = get_monitors()
    if monitor_idx >= len(monitors):
        print(f"Monitor index {monitor_idx} out of range.")
        return
    
    monitor = monitors[monitor_idx]
    window_name = f"Video_{monitor_idx}"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, monitor.x, monitor.y)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    wait_ms = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue
        
        # Resize frame to fit monitor
        frame_h, frame_w = frame.shape[:2]
        scale = min(monitor.width / frame_w, monitor.height / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Center on black background
        fullscreen_img = np.zeros((monitor.height, monitor.width, 3), dtype=np.uint8)
        y_offset = (monitor.height - new_h) // 2
        x_offset = (monitor.width - new_w) // 2
        fullscreen_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        cv2.imshow(window_name, fullscreen_img)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyWindow(window_name)


def spawn_fullscreen_videos_on_other_screens(video_paths: List[str]) -> None:
    """
    Spawn each video in video_paths in fullscreen on all available screens except the main one.
    If there are more screens than videos, videos will be reused in order.
    Requires 'screeninfo' package: pip install screeninfo
    """
    if get_monitors is None:
        print("screeninfo package not installed. Please install it: pip install screeninfo")
        return
    
    monitors = get_monitors()
    if len(monitors) < 2:
        print("Only one monitor detected. No additional screens to spawn videos on.")
        return
    
    processes = []
    for idx, monitor in enumerate(monitors[1:], start=1):  # Skip main monitor (index 0)
        video_path = video_paths[(idx-1) % len(video_paths)]
        p = multiprocessing.Process(target=_play_video_fullscreen_on_monitor, args=(video_path, idx))
        p.start()
        processes.append(p)
    
    print(f"Spawned {len(processes)} fullscreen video(s) on other screens.")
    # Optionally, join processes if you want to wait for them to finish
    # for p in processes:
    #     p.join()


def get_monitor_info() -> Optional[List]:
    """Get information about available monitors"""
    if get_monitors is None:
        return None
    
    try:
        monitors = get_monitors()
        monitor_info = []
        for i, monitor in enumerate(monitors):
            monitor_info.append({
                'index': i,
                'name': f"Monitor {i}",
                'x': monitor.x,
                'y': monitor.y,
                'width': monitor.width,
                'height': monitor.height,
                'is_primary': i == 0
            })
        return monitor_info
    except Exception as e:
        print(f"Error getting monitor info: {e}")
        return None


def create_fullscreen_window(window_name: str, monitor_idx: int = 0) -> bool:
    """Create a fullscreen window on the specified monitor"""
    if get_monitors is None:
        return False
    
    try:
        monitors = get_monitors()
        if monitor_idx >= len(monitors):
            print(f"Monitor index {monitor_idx} out of range.")
            return False
        
        monitor = monitors[monitor_idx]
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, monitor.x, monitor.y)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return True
    except Exception as e:
        print(f"Error creating fullscreen window: {e}")
        return False


def resize_frame_for_monitor(frame: np.ndarray, monitor_idx: int = 0) -> Optional[np.ndarray]:
    """Resize a frame to fit the specified monitor while preserving aspect ratio"""
    if get_monitors is None:
        return frame
    
    try:
        monitors = get_monitors()
        if monitor_idx >= len(monitors):
            return frame
        
        monitor = monitors[monitor_idx]
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate scale to fit monitor
        scale = min(monitor.width / frame_w, monitor.height / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Create fullscreen image with black background
        fullscreen_img = np.zeros((monitor.height, monitor.width, 3), dtype=np.uint8)
        y_offset = (monitor.height - new_h) // 2
        x_offset = (monitor.width - new_w) // 2
        fullscreen_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        return fullscreen_img
    except Exception as e:
        print(f"Error resizing frame: {e}")
        return frame 