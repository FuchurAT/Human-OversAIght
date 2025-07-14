import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import logging
from YOLOv8_Explainer import yolov8_heatmap

class GradCAM:
    def __init__(self, model, target_layer, model_path=None):
        self.model = model
        self.target_layer = target_layer
        self.model_path = model_path
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.temp_frame_path = "temp_frame.jpg"
        self.last_gradcam_img = None
        
        # Initialize YOLOv8 Explainer if model_path is provided
        if model_path:
            try:
                self.cam_model = yolov8_heatmap(
                    weight=model_path,
                    conf_threshold=0.15,
                    method="GradCAM",
                    show_box=True,
                    renormalize=False
                )
            except Exception as e:
                logging.warning(f"Failed to initialize YOLOv8 Explainer: {e}")
                self.cam_model = None
        else:
            self.cam_model = None
            
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        """Generate GradCAM for the input tensor"""
        if self.cam_model:
            # Use YOLOv8 Explainer if available
            return self._generate_with_explainer(input_tensor)
        else:
            # Use custom implementation
            return self._generate_custom(input_tensor, class_idx)

    def _generate_with_explainer(self, input_tensor):
        """Generate GradCAM using YOLOv8 Explainer"""
        try:
            # Convert tensor to image format
            if isinstance(input_tensor, torch.Tensor):
                # Convert tensor to numpy array
                if input_tensor.dim() == 4:
                    # Remove batch dimension and convert to image format
                    img_array = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    # Convert to 0-255 range
                    img_array = (img_array * 255).astype(np.uint8)
                    # Convert RGB to BGR for OpenCV
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_array = input_tensor.cpu().numpy()
            else:
                img_array = input_tensor

            # Save temporary frame
            cv2.imwrite(self.temp_frame_path, img_array)
            
            # Generate GradCAM
            cam_images = self.cam_model(img_path=self.temp_frame_path)
            gradcam_img = np.zeros_like(img_array)  # Default fallback
            
            if isinstance(cam_images, list) and len(cam_images) > 0:
                img_candidate = cam_images[0]
                if isinstance(img_candidate, Image.Image):
                    gradcam_img = np.array(img_candidate.convert("RGB"))
                    # Convert RGB to BGR for consistency
                    gradcam_img = cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR)
                elif isinstance(img_candidate, np.ndarray):
                    gradcam_img = img_candidate
                
                if isinstance(gradcam_img, np.ndarray) and gradcam_img.ndim >= 2:
                    if gradcam_img.shape[:2] != img_array.shape[:2]:
                        gradcam_img = cv2.resize(gradcam_img, (img_array.shape[1], img_array.shape[0]))
                else:
                    gradcam_img = np.zeros_like(img_array)
                    
            return gradcam_img
            
        except Exception as e:
            logging.warning(f"Grad-CAM generation with explainer failed: {e}")
            return np.zeros_like(input_tensor.cpu().numpy() if isinstance(input_tensor, torch.Tensor) else input_tensor)

    def _generate_custom(self, input_tensor, class_idx):
        """Generate GradCAM using custom implementation"""
        try:
            self.model.zero_grad()
            output = self.model(input_tensor)[0]  # YOLO returns a list
            
            if class_idx is None:
                # Use the highest confidence class
                class_idx = output.argmax(dim=1)
            
            score = output[:, class_idx].max()
            score.backward(retain_graph=True)
            
            gradients = self.gradients
            activations = self.activations
            
            if gradients is None or activations is None:
                return np.zeros_like(input_tensor.squeeze().cpu().numpy())
            
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Convert to heatmap
            cam_resized = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            
            return heatmap
            
        except Exception as e:
            logging.warning(f"Custom Grad-CAM generation failed: {e}")
            return np.zeros_like(input_tensor.squeeze().cpu().numpy())

    def create_final_frame(self, original_frame, gradcam_img, gradcam_enabled):
        """Create final frame with optional GradCAM overlay"""
        if not gradcam_enabled or gradcam_img is None:
            return original_frame
        
        try:
            # Ensure gradcam_img has the same shape as original_frame
            if gradcam_img.shape[:2] != original_frame.shape[:2]:
                gradcam_img = cv2.resize(gradcam_img, (original_frame.shape[1], original_frame.shape[0]))
            
            # Overlay heatmap on original frame
            final_frame = gradcam_img * 0.4 + original_frame * 0.6
            final_frame = final_frame.astype(np.uint8)
            
            return final_frame
            
        except Exception as e:
            logging.warning(f"Failed to create final frame: {e}")
            return original_frame

    def close(self):
        """Clean up hooks"""
        for handle in self.hook_handles:
            handle.remove()
