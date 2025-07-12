import cv2
import logging
import time

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