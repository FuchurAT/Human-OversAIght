"""
Bounding box management utilities for the object detection application.
"""

from typing import List, Tuple
from config.config import DEFAULT_IOU_THRESHOLD

from application.models import Detection


class BoxManager:
    """Handles bounding box operations and tracking"""
    
    def __init__(self, iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self.prev_boxes = []
        self.box_id_counter = 0
        self.box_id_lifetime = {}
        self.box_id_last_seen = {}
    
    def calculate_iou(self, box_a: Tuple[int, int, int, int], 
                     box_b: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        
        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        
        return inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)
    
    def lerp_box(self, box1: Tuple[int, int, int, int], 
                 box2: Tuple[int, int, int, int], alpha: float) -> Tuple[int, int, int, int]:
        """Linear interpolation between two boxes"""
        return tuple([int(a + (b - a) * alpha) for a, b in zip(box1, box2)])
    
    def filter_overlapping_boxes(self, detections: List[Detection]) -> List[Detection]:
        """Filter out overlapping detections, keeping highest confidence ones"""
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        selected_detections = []
        
        for detection in sorted_detections:
            overlap = False
            for selected in selected_detections:
                if self.calculate_iou(detection.box, selected.box) > self.iou_threshold:
                    overlap = True
                    break
            if not overlap:
                selected_detections.append(detection)
        
        return selected_detections
    
    def get_box_center(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get the center point of a bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_box_area(self, box: Tuple[int, int, int, int]) -> int:
        """Get the area of a bounding box"""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
    
    def expand_box(self, box: Tuple[int, int, int, int], 
                   expansion: int) -> Tuple[int, int, int, int]:
        """Expand a bounding box by a given amount"""
        x1, y1, x2, y2 = box
        return (x1 - expansion, y1 - expansion, x2 + expansion, y2 + expansion)
    
    def clip_box_to_frame(self, box: Tuple[int, int, int, int], 
                          frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Clip a bounding box to fit within frame boundaries"""
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        return (x1, y1, x2, y2) 