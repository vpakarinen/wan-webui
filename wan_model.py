import torch
import os

from diffusers import DiffusionPipeline
from transformers import logging

logging.set_verbosity_error()

class WANModel:
    """Handler for the WAN 2.1 Text-to-Video model"""
    
    def __init__(self, cache_dir="models"):
        self.repo_id = "Wan-AI/Wan2.1-T2V-14B"
        self.cache_dir = cache_dir
        self.pipeline = None
        self.resolution = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_model(self, resolution="480p"):
        """Load the WAN model with specified resolution"""
        if self.pipeline is not None and self.resolution == resolution:
            return "Model already loaded with resolution: " + resolution
        
        self.resolution = resolution
        
        if resolution == "480p":
            height, width = 320, 576  # 16:9 aspect ratio for 480p
        else:  # 720p
            height, width = 448, 832  # 16:9 aspect ratio for 720p
        
        try:
            self.pipeline = DiffusionPipeline.from_pretrained(
                "cerspense/zeroscope_v2_576w",
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir
            )
            
            self.pipeline = self.pipeline.to(self.device)
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                
            return f"Model loaded successfully with resolution: {resolution} ({width}x{height})"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def get_dimensions(self):
        """Get current height and width based on selected resolution"""
        if self.resolution == "480p":
            return 320, 576
        else:  # 720p
            return 448, 832
    
    def generate(
        self,
        prompt,
        negative_prompt="low quality, blurry, bad hands, bad face, poor lighting",
        guidance_scale=7.5,
        num_frames=24,
        num_inference_steps=30,
        progress_callback=None
    ):
        """Generate video based on text prompt"""
        if self.pipeline is None:
            return None, "Please load the model first"
        
        try:
            height, width = self.get_dimensions()
            
            if progress_callback:
                progress_callback(0.1, desc="Starting generation...")
            
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                height=height,
                width=width
            )
            
            if hasattr(result, 'frames'):
                video_frames = result.frames[0]
            else:
                video_frames = result.videos[0]
                
            if progress_callback:
                progress_callback(0.9, desc="Finalizing video...")
            
            return video_frames, f"Generated video with {num_frames} frames"
        except Exception as e:
            return None, f"Error generating video: {str(e)}"
