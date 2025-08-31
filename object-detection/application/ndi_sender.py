"""
NDI (Network Device Interface) sender for video frames.
Handles sending processed video frames to NDI-compatible receivers.
"""

import logging
import numpy as np
from typing import Optional, Tuple
from config.config import (
    DEFAULT_NDI_SOURCE_NAME, DEFAULT_NDI_GROUP_NAME, DEFAULT_NDI_VIDEO_FORMAT,
    DEFAULT_NDI_FRAME_RATE, DEFAULT_NDI_VIDEO_WIDTH, DEFAULT_NDI_VIDEO_HEIGHT
)


class NDISender:
    """Handles sending video frames via NDI"""
    
    def __init__(self, source_name: str = DEFAULT_NDI_SOURCE_NAME, 
                 group_name: str = DEFAULT_NDI_GROUP_NAME,
                 video_format: str = DEFAULT_NDI_VIDEO_FORMAT,
                 frame_rate: int = DEFAULT_NDI_FRAME_RATE,
                 video_width: int = DEFAULT_NDI_VIDEO_WIDTH,
                 video_height: int = DEFAULT_NDI_VIDEO_HEIGHT):
        """
        Initialize NDI sender
        
        Args:
            source_name: Name of the NDI source
            group_name: NDI group name (optional)
            video_format: Video format (BGRX, UYVY, RGBX, etc.)
            frame_rate: Target frame rate
            video_width: Video width in pixels
            video_height: Video height in pixels
        """
        self.source_name = source_name
        self.group_name = group_name
        self.video_format = video_format
        self.frame_rate = frame_rate
        self.video_width = video_width
        self.video_height = video_height
        
        # NDI objects
        self.ndi_send = None
        self.ndi_find = None
        self.ndi_recv = None
        
        # State
        self.is_initialized = False
        self.is_sending = False
        
        # Initialize NDI
        self._initialize_ndi()
    
    def _initialize_ndi(self) -> None:
        """Initialize NDI library and create sender"""
        try:
            import NDIlib as ndi
            
            # Initialize NDI
            if not ndi.initialize():
                logging.error("Failed to initialize NDI library")
                return
            
            # Create NDI finder
            self.ndi_find = ndi.find_create_v2()
            if not self.ndi_find:
                logging.error("Failed to create NDI finder")
                return
            
            # Create NDI sender
            send_create = ndi.SendCreate()
            send_create.ndi_name = self.source_name
            if self.group_name:
                send_create.ndi_group = self.group_name
            
            self.ndi_send = ndi.send_create(send_create)
            if not self.ndi_send:
                logging.error("Failed to create NDI sender")
                return
            
            self.is_initialized = True
            logging.info(f"NDI sender initialized successfully: {self.source_name}")
            
        except ImportError:
            logging.warning("NDI library not found. Install with: pip install NDI-Python")
            logging.warning("NDI functionality will be disabled")
        except Exception as e:
            logging.error(f"Failed to initialize NDI: {e}")
    
    def _convert_frame_format(self, frame: np.ndarray, target_format: str) -> np.ndarray:
        """
        Convert frame to target NDI format
        
        Args:
            frame: Input frame (BGR format)
            target_format: Target NDI format
            
        Returns:
            Converted frame
        """
        if frame is None:
            return None
        
        try:
            if target_format == "BGRX":
                # Convert BGR to BGRX (add alpha channel)
                if frame.shape[2] == 3:
                    # Add alpha channel
                    bgrx = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                    bgrx[:, :, :3] = frame
                    bgrx[:, :, 3] = 255  # Full alpha
                    return bgrx
                elif frame.shape[2] == 4:
                    return frame
                else:
                    logging.warning(f"Unexpected frame format: {frame.shape}")
                    return frame
            
            elif target_format == "RGBX":
                # Convert BGR to RGBX
                if frame.shape[2] == 3:
                    rgbx = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                    rgbx[:, :, 0] = frame[:, :, 2]  # B -> R
                    rgbx[:, :, 1] = frame[:, :, 1]  # G -> G
                    rgbx[:, :, 2] = frame[:, :, 0]  # R -> B
                    rgbx[:, :, 3] = 255  # Full alpha
                    return rgbx
                elif frame.shape[2] == 4:
                    # Convert BGRX to RGBX
                    rgbx = frame.copy()
                    rgbx[:, :, [0, 2]] = rgbx[:, :, [2, 0]]  # Swap R and B
                    return rgbx
                else:
                    logging.warning(f"Unexpected frame format: {frame.shape}")
                    return frame
            
            elif target_format == "UYVY":
                # Convert BGR to UYVY (YUV 4:2:2)
                if frame.shape[2] == 3:
                    # Convert BGR to YUV
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    # Convert to UYVY format
                    height, width = yuv.shape[:2]
                    uyvy = np.zeros((height, width // 2, 4), dtype=np.uint8)
                    
                    for i in range(0, width, 2):
                        uyvy[:, i // 2, 0] = yuv[:, i, 1]      # U
                        uyvy[:, i // 2, 1] = yuv[:, i, 0]      # Y1
                        uyvy[:, i // 2, 2] = yuv[:, i + 1, 1]  # V
                        uyvy[:, i // 2, 3] = yuv[:, i + 1, 0]  # Y2
                    
                    return uyvy
                else:
                    logging.warning(f"UYVY conversion requires 3-channel input, got: {frame.shape}")
                    return frame
            
            else:
                logging.warning(f"Unsupported NDI format: {target_format}, using BGRX")
                # Fallback to BGRX
                return self._convert_frame_format(frame, "BGRX")
                
        except Exception as e:
            logging.error(f"Frame format conversion failed: {e}")
            return frame
    
    def _resize_frame(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize frame to target dimensions
        
        Args:
            frame: Input frame
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized frame
        """
        if frame is None:
            return None
        
        try:
            import cv2
            current_height, current_width = frame.shape[:2]
            
            if current_width == target_width and current_height == target_height:
                return frame
            
            # Resize frame
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            return resized
            
        except ImportError:
            logging.warning("OpenCV not available for frame resizing")
            return frame
        except Exception as e:
            logging.error(f"Frame resizing failed: {e}")
            return frame
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send a frame via NDI
        
        Args:
            frame: Frame to send (BGR format)
            
        Returns:
            True if frame was sent successfully, False otherwise
        """
        if not self.is_initialized or self.ndi_send is None:
            return False
        
        if frame is None:
            logging.warning("Cannot send None frame")
            return False
        
        try:
            import NDIlib as ndi
            
            # Resize frame if needed
            resized_frame = self._resize_frame(frame, self.video_width, self.video_height)
            if resized_frame is None:
                return False
            
            # Convert to target format
            converted_frame = self._convert_frame_format(resized_frame, self.video_format)
            if converted_frame is None:
                return False
            
            # Create NDI video frame
            video_frame = ndi.VideoFrameV2()
            video_frame.xres = self.video_width
            video_frame.yres = self.video_height
            video_frame.FourCC = getattr(ndi, f"FOURCC_{self.video_format}")
            video_frame.frame_rate_N = self.frame_rate
            video_frame.frame_rate_D = 1
            video_frame.picture_aspect_ratio = self.video_width / self.video_height
            video_frame.frame_format_type = ndi.FRAME_FORMAT_TYPE_PROGRESSIVE
            video_frame.timecode = ndi.send_timecode_synthesize()
            video_frame.data = converted_frame.tobytes()
            video_frame.line_stride_in_bytes = converted_frame.strides[1]
            
            # Send frame
            ndi.send_send_video_v2(self.ndi_send, video_frame)
            
            self.is_sending = True
            return True
            
        except ImportError:
            logging.warning("NDI library not available")
            return False
        except Exception as e:
            logging.error(f"Failed to send NDI frame: {e}")
            return False
    
    def update_config(self, source_name: str = None, group_name: str = None,
                     video_format: str = None, frame_rate: int = None,
                     video_width: int = None, video_height: int = None) -> None:
        """
        Update NDI configuration
        
        Args:
            source_name: New source name
            group_name: New group name
            video_format: New video format
            frame_rate: New frame rate
            video_width: New video width
            video_height: New video height
        """
        if source_name is not None:
            self.source_name = source_name
        if group_name is not None:
            self.group_name = group_name
        if video_format is not None:
            self.video_format = video_format
        if frame_rate is not None:
            self.frame_rate = frame_rate
        if video_width is not None:
            self.video_width = video_width
        if video_height is not None:
            self.video_height = video_height
        
        logging.info(f"NDI configuration updated: {self.source_name}, {self.video_width}x{self.video_height}, {self.video_format}")
    
    def is_available(self) -> bool:
        """Check if NDI is available and initialized"""
        return self.is_initialized
    
    def cleanup(self) -> None:
        """Clean up NDI resources"""
        try:
            if self.ndi_send:
                import NDIlib as ndi
                ndi.send_destroy(self.ndi_send)
                self.ndi_send = None
            
            if self.ndi_find:
                import NDIlib as ndi
                ndi.find_destroy(self.ndi_find)
                self.ndi_find = None
            
            self.is_initialized = False
            self.is_sending = False
            logging.info("NDI resources cleaned up")
            
        except ImportError:
            pass
        except Exception as e:
            logging.warning(f"Error during NDI cleanup: {e}")


class NDISenderDummy:
    """Dummy NDI sender when NDI library is not available"""
    
    def __init__(self, *args, **kwargs):
        logging.warning("NDI library not available, using dummy sender")
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """Dummy send frame method"""
        return False
    
    def update_config(self, *args, **kwargs) -> None:
        """Dummy update config method"""
        pass
    
    def is_available(self) -> bool:
        """Always returns False for dummy sender"""
        return False
    
    def cleanup(self) -> None:
        """Dummy cleanup method"""
        pass


# Factory function to create appropriate NDI sender
def create_ndi_sender(*args, **kwargs):
    """Create NDI sender or dummy if NDI not available"""
    try:
        import NDIlib
        return NDISender(*args, **kwargs)
    except ImportError:
        return NDISenderDummy(*args, **kwargs)
