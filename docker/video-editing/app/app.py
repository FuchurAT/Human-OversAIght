from pydantic import BaseModel
import cv2
import torch
import numpy as np
from PIL import Image
import os
import time
from tqdm import tqdm
import uuid
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException

# Request/Response models
class VideoInpaintRequest(BaseModel):
    video_path: str
    prompt: str
    source_object: Optional[str] = "car"
    target_object: Optional[str] = "school bus"
    strength: Optional[float] = 0.7
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 20
    debug: Optional[bool] = False  # Added debug parameter

class VideoInpaintResponse(BaseModel):
    output_video_path: str
    generation_time: float
    status: str
    message: str
    video_duration: float
    total_frames: int

class VideoInpaintServer:
    def __init__(self, device="cuda", output_dir="../../../data/output"): #change to /data/output
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Load models
        self.load_models()
        
        print(f"Server initialized with device: {device}")
        print(f"Output directory: {self.output_dir}")
    
    def load_models(self):
        """Load all required models"""
        print("Loading models...")
        start_time = time.time()
        
        try:
            # Load inpainting model
            from diffusers import StableDiffusionInpaintPipeline
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
            
            # Memory optimizations
            self.inpaint_pipe.enable_model_cpu_offload()
            self.inpaint_pipe.enable_attention_slicing()
            
            # Load object detection
            try:
                import ultralytics
                self.yolo_model = ultralytics.YOLO('models/object-detection.pt')
                print("✓ YOLO model loaded")
            except ImportError:
                from transformers import pipeline
                self.detector = pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",
                    device=0 if self.device == "cuda" else -1
                )
                print("✓ DETR model loaded (fallback)")
            
            # Load SAM for precise segmentation
            try:
                from segment_anything import sam_model_registry, SamPredictor
                sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
                self.sam_predictor = SamPredictor(sam.to(self.device))
                print("✓ SAM model loaded")
            except ImportError:
                print("⚠ SAM not available, using basic masking")
                self.sam_predictor = None
            
            load_time = time.time() - start_time
            print(f"✓ All models loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def detect_and_track_object(self, frames, target_object):
        """Detect and track object across frames"""
        object_tracks = []
        
        for i, frame in enumerate(frames):
            bbox = None
            detected_classes = []  # Debug: store detected class names
            
            if hasattr(self, 'yolo_model'):
                # Use YOLO
                results = self.yolo_model(frame, verbose=False)
                for result in results:
                    for box in result.boxes:
                        class_name = self.yolo_model.names[int(box.cls)]
                        detected_classes.append(class_name)  # Debug
                        if target_object.lower() in class_name.lower():
                            bbox = box.xyxy[0].cpu().numpy()
                            break
            else:
                # Use DETR fallback
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                detections = self.detector(pil_image)
                
                for detection in detections:
                    detected_classes.append(detection['label'])  # Debug
                    if target_object.lower() in detection['label'].lower():
                        bbox = [
                            detection['box']['xmin'],
                            detection['box']['ymin'],
                            detection['box']['xmax'],
                            detection['box']['ymax']
                        ]
                        break
            # Debug print for detected classes and bbox
            print(f"[Frame {i}] Detected classes: {detected_classes}, Selected bbox: {bbox}")
            
            object_tracks.append({
                'frame_idx': i,
                'bbox': bbox,
                'confidence': 0.8 if bbox is not None else 0
            })
        
        return self.smooth_tracking(object_tracks)
    
    def smooth_tracking(self, tracks):
        """Smooth tracking by interpolating missing detections"""
        for i in range(len(tracks)):
            if tracks[i]['bbox'] is None:
                prev_valid = None
                next_valid = None
                
                # Find previous valid detection
                for j in range(i-1, -1, -1):
                    if tracks[j]['bbox'] is not None:
                        prev_valid = tracks[j]
                        break
                
                # Find next valid detection
                for j in range(i+1, len(tracks)):
                    if tracks[j]['bbox'] is not None:
                        next_valid = tracks[j]
                        break
                
                # Interpolate if both exist
                if prev_valid and next_valid:
                    alpha = 0.5
                    prev_bbox = np.array(prev_valid['bbox'])
                    next_bbox = np.array(next_valid['bbox'])
                    tracks[i]['bbox'] = (prev_bbox * (1-alpha) + next_bbox * alpha).tolist()
        
        return tracks
    
    def create_masks(self, frames, object_tracks):
        """Create masks for each frame"""
        masks = []
        
        for i, track in enumerate(object_tracks):
            frame = frames[i]
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            if track['bbox'] is not None:
                x1, y1, x2, y2 = map(int, track['bbox'])
                
                if self.sam_predictor:
                    # Use SAM for precise segmentation
                    self.sam_predictor.set_image(frame)
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    masks_sam, scores, logits = self.sam_predictor.predict(
                        point_coords=np.array([[center_x, center_y]]),
                        point_labels=np.array([1])
                    )
                    
                    best_mask = masks_sam[np.argmax(scores)]
                    mask = (best_mask * 255).astype(np.uint8)
                else:
                    # Simple bounding box mask
                    mask[y1:y2, x1:x2] = 255


            if mask.sum() != 0:
                # Debug print for mask sum
                print(f"[Frame {i}] Mask sum: {mask.sum()}")
            
            masks.append(mask)
        
        return masks
    
    def process_video_inpainting(self, frames, masks, prompt, **kwargs):
        """Process video with inpainting"""
        output_frames = []
        prev_result = None
        
        strength = kwargs.get('strength', 0.7)
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        num_inference_steps = kwargs.get('num_inference_steps', 20)
        
        for i, (frame, mask) in enumerate(tqdm(zip(frames, masks), desc="Processing frames")):
            # Skip if no mask
            if mask.sum() == 0:
                output_frames.append(frame)
                continue
            
            # Convert to PIL
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(mask)
            
            # Use previous result for consistency
            init_image = prev_result if prev_result is not None else pil_frame
            
            try:
                result = self.inpaint_pipe(
                    prompt=prompt,
                    image=init_image,
                    mask_image=pil_mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                ).images[0]
                
                # Blend with original for consistency
                result_array = np.array(result)
                original_array = np.array(pil_frame)
                mask_3d = np.stack([mask/255.0] * 3, axis=-1)
                
                # Smooth blending
                blended = (result_array * mask_3d + original_array * (1 - mask_3d)).astype(np.uint8)
                
                output_frames.append(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                prev_result = Image.fromarray(blended)
                
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                output_frames.append(frame)  # Use original frame on error
            
            # Clear GPU memory periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        return output_frames
    
    def extract_frames(self, video_path):
        """Extract frames from video"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, fps, duration, total_frames
    
    def frames_to_video(self, frames, output_path, fps):
        """Convert frames to video"""
        if not frames:
            raise ValueError("No frames to convert")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    async def process_video_async(self, request: VideoInpaintRequest):
        """Process video asynchronously"""
        start_time = time.time()
        
        try:
            # Generate unique output filename
            output_filename = f"output_{uuid.uuid4().hex}.mp4"
            output_path = self.output_dir / output_filename
            
            # Extract frames
            frames, fps, duration, total_frames = self.extract_frames(request.video_path)
            
            # Debug: Only process first 100 frames if debug is True
            if getattr(request, 'debug', False):
                frames = frames[:100]
                total_frames = len(frames)
                duration = total_frames / fps if fps > 0 else 0
            
            # Detect and track object
            object_tracks = self.detect_and_track_object(frames, request.source_object)
            
            # Create masks
            masks = self.create_masks(frames, object_tracks)
            
            # Process with inpainting
            output_frames = self.process_video_inpainting(
                frames, masks, request.prompt,
                strength=request.strength,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps
            )
            
            # Save video
            self.frames_to_video(output_frames, output_path, fps)
            
            generation_time = time.time() - start_time
            
            return VideoInpaintResponse(
                output_video_path=str(output_path),
                generation_time=generation_time,
                status="success",
                message="Video processing completed successfully",
                video_duration=duration,
                total_frames=total_frames
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            return VideoInpaintResponse(
                output_video_path="",
                generation_time=generation_time,
                status="error",
                message=str(e),
                video_duration=0,
                total_frames=0
            )

# FastAPI app
app = FastAPI(
    title="Video Object Inpainting API",
    description="AI-powered video object replacement using inpainting",
    version="1.0.0"
)

# Initialize server
server = VideoInpaintServer()

@app.get("/inpaint-video", response_model=VideoInpaintResponse)
async def inpaint_video(
    video_path: str,
    prompt: str,
    source_object: Optional[str] = "car",
    target_object: Optional[str] = "school bus",
    strength: Optional[float] = 0.7,
    guidance_scale: Optional[float] = 7.5,
    num_inference_steps: Optional[int] = 20,
    debug: Optional[bool] = False  # Added debug parameter
):
    request = VideoInpaintRequest(
        video_path=video_path,
        prompt=prompt,
        source_object=source_object,
        target_object=target_object,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        debug=debug
    )
    try:
        result = await server.process_video_async(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": server.device}

@app.get("/")
async def root():
    """Root endpoint with usage instructions"""
    return {
        "message": "Video Object Inpainting API",
        "usage": {
            "endpoint": "/inpaint-video",
            "method": "GET",
            "example": {
                "video_path": "/path/to/video.mp4",
                "prompt": "realistic yellow school bus",
                "source_object": "tank",
                "target_object": "school bus"
            }
        }
    }