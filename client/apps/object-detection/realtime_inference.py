import cv2
import logging
from pathlib import Path
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from YOLOv8_Explainer import yolov8_heatmap
from PIL import Image
import torch
from skimage.draw import line as skline

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
    "RIOT_POLICE",
    "TEARGAS"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DetectionVisualizer:
    def __init__(self, classes, app_instance=None):
        self.classes = classes
        self.app_instance = app_instance
        self.prev_boxes = []
        self.box_id_counter = 0
        self.box_id_lifetime = {}
        self.box_id_last_seen = {}
        self.frame_idx = 0
        self.iou_threshold = 0.2
        self.animation_alpha = 0.3
        self.color_state = 'blue'
        self.show_enemy = False
        self.solid_border = False
        self.blur_boxes = True
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_fps_text = "FPS: 0.0"
        self.last_inf_text = "Inference: 0.0 ms"

    def confidence_to_color(self, conf):
        # conf: 0.0 (orange) to 1.0 (yellow)
        # Orange: (0, 165, 255), Yellow: (0, 255, 255) in BGR
        # Interpolate green channel from 165 (orange) to 255 (yellow)
        b = 255
        g = int(165 + (255 - 105) * conf)
        r = 0
        return (r, g, b)

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def lerp_box(self, box1, box2, alpha):
        return tuple([int(a + (b - a) * alpha) for a, b in zip(box1, box2)])

    def draw_detection_overlays(self, frame, overlay_boxes, legend_dict):
        # Draw all detection overlays (boxes, corners, glow, ENEMY text) on the frame
        for box, conf, class_id, anim_box, pid in overlay_boxes:
            if self.color_state == 'red':
                color = (255, 0, 0)  # Red in BGR
            else:
                color = self.confidence_to_color(conf)
            if self.solid_border:
                cv2.rectangle(frame, (anim_box[0], anim_box[1]), (anim_box[2], anim_box[3]), color, 3)
            else:
                x1, y1, x2, y2 = anim_box
                corner_length = 20
                thickness = 3
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
                # Top-right
                cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
                # Bottom-left
                cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
                # Middle of each edge
                mid_length = 20
                # Top edge
                mid_top_x1 = x1 + (x2 - x1) // 2 - mid_length // 2
                mid_top_x2 = mid_top_x1 + mid_length
                mid_top_y = y1
                cv2.line(frame, (mid_top_x1, mid_top_y), (mid_top_x2, mid_top_y), color, thickness)
                # Bottom edge
                mid_bot_x1 = x1 + (x2 - x1) // 2 - mid_length // 2
                mid_bot_x2 = mid_bot_x1 + mid_length
                mid_bot_y = y2
                cv2.line(frame, (mid_bot_x1, mid_bot_y), (mid_bot_x2, mid_bot_y), color, thickness)
                # Left edge
                mid_left_y1 = y1 + (y2 - y1) // 2 - mid_length // 2
                mid_left_y2 = mid_left_y1 + mid_length
                mid_left_x = x1
                cv2.line(frame, (mid_left_x, mid_left_y1), (mid_left_x, mid_left_y2), color, thickness)
                # Right edge
                mid_right_y1 = y1 + (y2 - y1) // 2 - mid_length // 2
                mid_right_y2 = mid_right_y1 + mid_length
                mid_right_x = x2
                cv2.line(frame, (mid_right_x, mid_right_y1), (mid_right_x, mid_right_y2), color, thickness)

                if self.blur_boxes:
                    roi = frame[y1:y2, x1:x2]
                    # Pixelate: downscale and then upscale
                    h, w = roi.shape[:2]
                    if h > 0 and w > 0:
                        pixel_size = max(4, min(h, w) // 8)  # adjust pixel_size as needed
                        temp = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                        frame[y1:y2, x1:x2] = pixelated

            # Glow effect
            overlay = frame.copy()
            glow_color = (0, 255, 255) if self.color_state != 'red' else (255, 0, 0)
            cv2.rectangle(overlay, (anim_box[0], anim_box[1]), (anim_box[2], anim_box[3]), glow_color, 12)
            alpha = 0.2
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # ENEMY text
            if self.show_enemy:
                cv2.putText(frame, "ENEMY", (anim_box[0], anim_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def draw_legend(self, frame, legend_dict):
        legend_lines = []
        legend_colors = []
        for class_id, (conf, color) in legend_dict.items():
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"Class_{class_id}"
            legend_lines.append(f"{class_name}: {conf:.2f}")
            legend_colors.append(color)
        if legend_lines:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            line_height = 25
            padding = 10
            box_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in legend_lines]) + 2 * padding
            box_height = line_height * len(legend_lines) + 2 * padding
            frame_h, frame_w = frame.shape[:2]
            box_x1 = frame_w - box_width - 10
            box_y1 = 10
            box_x2 = frame_w - 10
            box_y2 = box_y1 + box_height
            # Draw filled rectangle
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (30, 30, 30), -1)
            # Draw text lines
            for idx, line in enumerate(legend_lines):
                text_x = box_x1 + padding
                text_y = box_y1 + padding + (idx + 1) * line_height - 7
                lcolor = legend_colors[idx]
                cv2.putText(frame, line, (text_x, text_y), font, font_scale, lcolor, font_thickness, cv2.LINE_AA)

    def draw_fps_info(self, frame, inf_time):
        # Draw FPS and inference time (update every 10 frames for performance)
        self.frame_count += 1
        if self.frame_idx % 10 == 0:
            now = time.time()
            if now - self.last_fps_time > 1.0:
                self.current_fps = self.frame_count / (now - self.last_fps_time)
                self.last_fps_time = now
                self.frame_count = 0
            fps_text = f"FPS: {self.current_fps:.1f}"
            inf_text = f"Inference: {inf_time:.1f} ms"
            # Store for reuse
            self.last_fps_text = fps_text
            self.last_inf_text = inf_text
        else:
            # Reuse last calculated values
            fps_text = self.last_fps_text
            inf_text = self.last_inf_text
        cv2.putText(frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, inf_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    def process_detections(self, detections, box_threshold):
        # Filter out overlapping boxes
        filtered = []
        for det in detections:
            boxA = det[0]
            overlap = False
            for boxB, _, _ in filtered:
                if self.iou(boxA, boxB) > self.iou_threshold:
                    overlap = True
                    break
            if not overlap:
                filtered.append(det)
        # Track boxes for animation and highlighting
        new_prev_boxes = []
        matched_prev_ids = set()
        legend_dict = {}  # class_id: (conf, color)
        overlay_boxes = []  # (box, conf, class_id, anim_box, pid)
        for box, conf, class_id in filtered:
            if conf >= box_threshold:
                if class_id not in legend_dict or conf > legend_dict[class_id][0]:
                    if self.color_state == 'red':
                        color = (255, 0, 0)
                    else:
                        color = self.confidence_to_color(conf)
                    legend_dict[class_id] = (conf, color)
            # Try to match with previous boxes (by IoU and class)
            best_iou = 0
            best_idx = -1
            for idx, (pbox, pclass, pid) in enumerate(self.prev_boxes):
                if class_id == pclass:
                    iou_val = self.iou(box, pbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = idx
            if best_iou > 0.3 and best_idx != -1:
                # Matched: animate
                pbox, pclass, pid = self.prev_boxes[best_idx]
                anim_box = self.lerp_box(pbox, box, self.animation_alpha)
                new_prev_boxes.append((box, class_id, pid))
                matched_prev_ids.add(pid)
            else:
                # New object
                self.box_id_counter += 1
                pid = self.box_id_counter
                anim_box = box
                new_prev_boxes.append((box, class_id, pid))
                self.box_id_lifetime[pid] = 0
            overlay_boxes.append((box, conf, class_id, anim_box, pid))
            self.box_id_last_seen[pid] = self.frame_idx
        # Update lifetimes
        for _, _, pid in new_prev_boxes:
            self.box_id_lifetime[pid] = self.box_id_lifetime.get(pid, 0) + 1
        # Remove old boxes
        self.prev_boxes = [b for b in new_prev_boxes if self.box_id_last_seen.get(b[2], self.frame_idx) == self.frame_idx]
        return overlay_boxes, legend_dict

    def handle_key_press(self, key):
        if key == 32:  # Space bar
            self.blur_boxes = not self.blur_boxes
            if self.color_state == 'red':
                self.color_state = 'yellow_orange'
            else:
                self.color_state = 'red'
            self.show_enemy = not self.show_enemy
            self.solid_border = not self.solid_border
        elif key == ord('g'):  # 'g' key to toggle Grad-CAM
            if self.app_instance:
                self.app_instance.gradcam_enabled = not self.app_instance.gradcam_enabled
                logging.info(f"Grad-CAM {'enabled' if self.app_instance.gradcam_enabled else 'disabled'}")

class VideoInferenceApp:
    def __init__(self, video_path, model_path, output_path, box_threshold=0.1):
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.box_threshold = box_threshold
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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)[0]  # YOLO returns a list
        score = output[:, class_idx].max()
        score.backward(retain_graph=True)
        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        for handle in self.hook_handles:
            handle.remove()

def main():
    parser = argparse.ArgumentParser(description="Run real-time inference on a video file with YOLO.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output", type=str, default="", help="Path to save the output video") # "output/realtime_detection.mp4"
    parser.add_argument("--box-threshold", type=float, default=0.1, help="Minimum confidence for showing a detection box")
    args = parser.parse_args()
 
    model_path = "E:/Projects/human-oversaight/client/apps/object-detection/runs/train/weights/best.pt"
    
    try:
        app = VideoInferenceApp(args.video_path, model_path, args.output, args.box_threshold)
        app.run()
    except Exception as e:
        logging.error(f"Error running inference: {e}")
        return
 
if __name__ == "__main__":
    main() 