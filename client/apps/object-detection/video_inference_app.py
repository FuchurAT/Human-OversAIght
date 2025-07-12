import cv2
import logging
from pathlib import Path
import time
import numpy as np
import torch
from YOLOv8_Explainer import yolov8_heatmap
from PIL import Image
from detection_visualizer import DetectionVisualizer

# Use the same class list as in inference_armored_vehicles.py
CLASSES = [
    "ARMED_POLICEMEN",
    "CAR_FIRE",
    "FIRE",
    "FIRE_FIREFIGHTER",
    "FIRE_TRUCK",
    "HEALTH_AMBULANCE",
    "IMMIGRANT",
    "MILITARY_OFFICER",
    "MILITARY_SOLDIER",
    "MILITARY_VIECHLE",
    "POLICE",
    "POLICE_CAR",
    "POLICE_TRUCK",
    "POLICEMAN",
    "PRISON",
    "PROTEST",
    "RIOT",
    "RIOT_POLICE"
]

class VideoInferenceApp:
    def __init__(self, video_path, model_path, output_path, box_threshold=0.1, show_legend=False):
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.box_threshold = box_threshold
        self.show_legend = show_legend
        self.temp_frame_path = "temp_frame.jpg"
        self.last_gradcam_img = None
        self.gradcam_enabled = False
        
        # Initialize model
        try:
            from ultralytics import YOLO
        except ImportError:
            logging.error("ultralytics package not found. Please install it: pip install ultralytics")
            raise
        
        if not Path(model_path).exists():
            logging.error(f"Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not Path(video_path).exists():
            logging.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Add this import
        from torch.nn.modules.container import Sequential, ModuleList
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.batchnorm import BatchNorm2d
        from torch.nn.modules.activation import SiLU
        from torch.nn.modules.pooling import MaxPool2d
        from torch.nn.modules.upsampling import Upsample
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.conv import Conv, Concat
        from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck, DFL
        from ultralytics.nn.modules.head import Detect

        # Add this line BEFORE loading the model
        torch.serialization.add_safe_globals([
            DetectionModel, Detect, Sequential, Conv, Conv2d, MaxPool2d, BatchNorm2d, SiLU, C2f, Bottleneck, DFL, ModuleList, SPPF, Upsample, Concat
        ])

        self.model = YOLO(model_path)
        #torch.serialization.add_safe_globals([self.model])
        #torch.serialization.safe_globals([self.model])
        logging.info(f"Loaded model: {model_path}")
        
        # Initialize Grad-CAM explainer
        self.cam_model = yolov8_heatmap(
            weight=model_path,
            conf_threshold=0.15,
            method="GradCAM",
            show_box=True,
            renormalize=False
        )
        
        # Initialize visualizer with reference to this app instance
        self.visualizer = DetectionVisualizer(CLASSES, self)
        
    def get_gradcam_image(self, frame):
        try:
            cv2.imwrite(self.temp_frame_path, frame)
            cam_images = self.cam_model(img_path=self.temp_frame_path)
            gradcam_img = np.zeros_like(frame)  # Default fallback
            
            if isinstance(cam_images, list) and len(cam_images) > 0:
                img_candidate = cam_images[0]
                if isinstance(img_candidate, Image.Image):
                    gradcam_img = np.array(img_candidate.convert("RGB"))
                elif isinstance(img_candidate, np.ndarray):
                    gradcam_img = img_candidate
                
                if isinstance(gradcam_img, np.ndarray) and gradcam_img.ndim >= 2:
                    if gradcam_img.shape[:2] != frame.shape[:2]:
                        gradcam_img = cv2.resize(gradcam_img, (frame.shape[1], frame.shape[0]))
                else:
                    gradcam_img = np.zeros_like(frame)
        except Exception as e:
            logging.warning(f"Grad-CAM generation failed: {e}")
            gradcam_img = np.zeros_like(frame)
        return gradcam_img
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        wait_ms = int(1000 / fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if(self.output_path != ""):
            # Prepare output video writer
            output_dir = str(Path(self.output_path).parent)
            if not Path(output_dir).exists():
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            logging.info(f"Saving output video to: {self.output_path}")
        

        logging.info("Press SPACE to toggle border color and show 'ENEMY'. Press 'q' or ESC to quit.")

        # Make OpenCV window full screen
        window_name = 'Object detection'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # --- Fullscreen window size detection ---
        fullscreen_size = None
        first_frame = True

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video or failed to grab frame.")
                break
            # Run inference
            inf_start = time.time()
            results = self.model(frame, conf=0.25)
            inf_time = (time.time() - inf_start) * 1000  # ms
            boxes = results[0].boxes
            detections = []  # (box, conf, class_id)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    if conf < self.box_threshold:
                        continue
                    class_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append((tuple(xyxy), conf, class_id))
            
            # Grad-CAM for all non-overlapping, highest-confidence detections
            gradcam_img = np.zeros_like(frame)  # Default initialization
            if self.visualizer.frame_idx % 5 == 0:
                try:
                    # Sort detections by confidence, descending
                    sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)
                    selected_boxes = []
                    selected_confs = []
                    selected_classes = []
                    for box, conf, class_id in sorted_detections:
                        # Check overlap with already selected boxes
                        overlap = False
                        for sel_box in selected_boxes:
                            # Use the same IoU threshold as visualizer
                            if self.visualizer.iou(box, sel_box) > self.visualizer.iou_threshold:
                                overlap = True
                                break
                        if not overlap:
                            selected_boxes.append(box)
                            selected_confs.append(conf)
                            selected_classes.append(class_id)
                    # Overlay Grad-CAM for each selected box
                    gradcam_img = np.zeros_like(frame)
                    for box, conf, class_id in zip(selected_boxes, selected_confs, selected_classes):
                        # Optionally, you could crop the frame to the box, but here we use the whole frame
                        single_gradcam = self.get_gradcam_image(frame)
                        # Overlay: simple max for heatmap effect
                        gradcam_img = np.maximum(gradcam_img, single_gradcam)
                    self.last_gradcam_img = gradcam_img
                except Exception as e:
                    logging.warning(f"Grad-CAM failed on frame {self.visualizer.frame_idx}: {e}")
                    gradcam_img = self.last_gradcam_img if self.last_gradcam_img is not None else np.zeros_like(frame)
            else:
                gradcam_img = self.last_gradcam_img if self.last_gradcam_img is not None else np.zeros_like(frame)
            
            # Select non-overlapping, highest-confidence boxes (same as Grad-CAM logic)
            sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)
            selected_boxes = []
            selected_confs = []
            selected_classes = []
            for box, conf, class_id in sorted_detections:
                overlap = False
                for sel_box in selected_boxes:
                    if self.visualizer.iou(box, sel_box) > self.visualizer.iou_threshold:
                        overlap = True
                        break
                if not overlap:
                    selected_boxes.append(box)
                    selected_confs.append(conf)
                    selected_classes.append(class_id)
            # Build overlay_boxes and legend_dict for visualizer
            overlay_boxes = []
            legend_dict = {}
            for box, conf, class_id in zip(selected_boxes, selected_confs, selected_classes):
                pid = 0  # No tracking needed for overlays
                anim_box = box
                overlay_boxes.append((box, conf, class_id, anim_box, pid))
                if class_id not in legend_dict or conf > legend_dict[class_id][0]:
                    if self.visualizer.color_state == 'red':
                        color = (255, 0, 0)
                    else:
                        color = self.visualizer.confidence_to_color(conf)
                    legend_dict[class_id] = (conf, color)

            # --- Prepare frames for display and output ---
            # Copy original frame for overlays
            frame_with_overlays = frame.copy()
            self.visualizer.draw_detection_overlays(frame_with_overlays, overlay_boxes, legend_dict)
            if self.show_legend:
                self.visualizer.draw_legend(frame_with_overlays, legend_dict)
            self.visualizer.draw_fps_info(frame_with_overlays, inf_time)

            # --- Fast display logic ---
            # Reduce frame size for display (e.g., 640 width)
            display_width = 640
            frame_rgb = frame_with_overlays[..., ::-1]
            h, w = frame_rgb.shape[:2]
            scale = display_width / w
            display_h = int(h * scale)
            frame_disp = np.array(Image.fromarray(frame_rgb).resize((display_width, display_h)))
            # Show only one view depending on gradcam_enabled
            if self.gradcam_enabled:
                try:
                    gradcam_rgb = gradcam_img[..., ::-1]
                    gradcam_disp = np.array(Image.fromarray(gradcam_rgb).resize((display_width, display_h)))
                    display_img = gradcam_disp
                except Exception as e:
                    logging.warning(f"Grad-CAM display failed: {e}")
                    display_img = frame_disp
            else:
                display_img = frame_disp

            # --- Fullscreen aspect-ratio preserving display ---
            if first_frame:
                # Show a dummy image to force window to full screen
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow(window_name, dummy)
                cv2.waitKey(1)
                # Try to get real window size
                try:
                    wx, wy, ww, wh = cv2.getWindowImageRect(window_name)
                except Exception:
                    ww, wh = 0, 0
                # If size is too small, use tkinter as fallback
                if ww < 300 or wh < 300:
                    try:
                        import tkinter as tk
                        root = tk.Tk()
                        ww = root.winfo_screenwidth()
                        wh = root.winfo_screenheight()
                        root.destroy()
                    except Exception:
                        ww, wh = 1920, 1080
                fullscreen_size = (ww, wh)
                first_frame = False
            else:
                ww, wh = fullscreen_size
            ih, iw = display_img.shape[:2]
            scale = min(ww / iw, wh / ih)
            new_w, new_h = int(iw * scale), int(ih * scale)
            resized_img = cv2.resize(display_img, (new_w, new_h))
            # Center on black background
            fullscreen_img = np.zeros((wh, ww, 3), dtype=np.uint8)
            y_offset = (wh - new_h) // 2
            x_offset = (ww - new_w) // 2
            fullscreen_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
            cv2.imshow(window_name, fullscreen_img)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q') or key == 27:
                break
            else:
                self.visualizer.handle_key_press(key)
            
            # --- Write to output video ---
            if self.gradcam_enabled:
                # Write Grad-CAM image (no overlays) in BGR
                out_frame = gradcam_img.copy()
                if out_frame.shape[2] == 3 and self.output_path != "":
                    out_writer.write(out_frame)
            elif self.output_path != "":
                # Write frame with overlays in BGR
                out_writer.write(frame_with_overlays)

            self.visualizer.frame_idx += 1
        cap.release()

        if self.output_path != "":
            out_writer.release()

        cv2.destroyAllWindows() 