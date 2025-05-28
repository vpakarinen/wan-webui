import subprocess
import traceback
import tempfile
import logging
import torch
import sys
import os

logger = logging.getLogger("videogen-webui")

possible_locations = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2.1"),       
    os.path.join("/workspace", "wan-webui", "Wan2.1"),                         
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Wan2.1")
]

WAN_DIR = None
for location in possible_locations:
    if os.path.exists(location) and os.path.exists(os.path.join(location, "generate.py")):
        WAN_DIR = location
        logger.info(f"Found WAN repository at: {WAN_DIR}")
        break

if WAN_DIR is None:
    logger.warning("WAN repository not found in any of the expected locations")

if WAN_DIR is not None:
    logger.info(f"Found WAN repository at {WAN_DIR}")
else:
    logger.warning("WAN repository not found in any expected location")
    logger.info("Will need to clone the WAN repository")

class WANModel:
    """Handler for the WAN 2.1 Text-to-Video model using the official WAN repository"""
    
    def __init__(self, cache_dir="models"):
        self.model_id = "Wan-AI/Wan2.1-T2V-14B"
        self.cache_dir = cache_dir
        self.resolution = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        os.makedirs(cache_dir, exist_ok=True)
        
        if WAN_DIR is not None:
            self.wan_dir = WAN_DIR
            self.generate_script = os.path.join(self.wan_dir, "generate.py")
            logger.info(f"Using WAN repository at: {self.wan_dir}")
        else:
            logger.info("WAN repository not found. Attempting to clone it...")
            try:
                self.wan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2.1")
                if not os.path.exists(self.wan_dir):
                    subprocess.run(
                        ["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", self.wan_dir],
                        check=True
                    )
                    logger.info(f"Successfully cloned WAN repository to {self.wan_dir}")
                else:
                    logger.info(f"WAN directory exists at {self.wan_dir} but generate.py wasn't found earlier")
                
                if self.wan_dir not in sys.path:
                    sys.path.insert(0, self.wan_dir)
                    logger.info(f"Added {self.wan_dir} to Python path")
                
                self.generate_script = os.path.join(self.wan_dir, "generate.py")
            except Exception as e:
                logger.error(f"Failed to clone WAN repository: {str(e)}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to clone WAN repository: {str(e)}")
        
        if not os.path.exists(self.generate_script):
            logger.error(f"WAN generate script not found at {self.generate_script}")
            raise FileNotFoundError(f"WAN generate script not found at {self.generate_script}")
        
        logger.info(f"WAN generate script found at {self.generate_script}")
    
    def load_model(self, resolution="480p"):
        """Just set the resolution for the WAN model (no actual loading needed)"""
        self.resolution = resolution
        
        if resolution == "480p":
            width, height = 832, 480  # Standard WAN 480p resolution
        else:
            width, height = 1280, 720  # Standard WAN 720p resolution
        
        logger.info(f"WAN model will use resolution: {resolution} ({width}x{height})")
        return f"Ready to generate videos at resolution: {resolution} ({width}x{height})"
    
    def get_dimensions(self):
        """Get current width and height based on selected resolution"""
        if self.resolution == "480p":
            return 832, 480  # Standard WAN 480p resolution (width, height)
        else:
            return 1280, 720  # Standard WAN 720p resolution (width, height)
    
    def generate(self, prompt, height=None, width=None, num_inference_steps=25, guidance_scale=7.5, progress_callback=None, negative_prompt="", motion_bucket_id=127):
        """Generate a video based on the text prompt using the WAN model via subprocess"""
        if height is None or width is None:
            width, height = self.get_dimensions()
        
        size = f"{width}*{height}"
        
        logger.info(f"Generating video with WAN model using prompt: {prompt}")
        logger.info(f"Parameters: size={size}, guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}")
        
        try:
            temp_dir = tempfile.mkdtemp()
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = os.path.join(output_dir, "output.mp4")
            my_env = os.environ.copy()
            task = "t2v-14B"

            logger.info(f"Using resolution {self.resolution} ({size})")

            cmd = [
                sys.executable,  # Python executable
                self.generate_script,  # generate.py script
                "--task", task,  # Use the appropriate model variant based on resolution
                "--size", size,  # Width*Height
                "--ckpt_dir", self.cache_dir,  # Model checkpoint directory
                "--prompt", prompt,  # Text prompt
                "--sample_steps", str(num_inference_steps),  # Number of inference steps (25 is the default)
                "--sample_guide_scale", str(guidance_scale),  # Guidance scale (7.5 is the default)
                "--sample_shift", str(motion_bucket_id / 127.0),  # Convert motion_bucket_id to sample_shift (0-1 range)
                "--save_file", output_filename  # Output video file
            ]
            
            if negative_prompt:
                cmd.extend(["--n_prompt", negative_prompt])
            
            if progress_callback is not None:
                progress_callback(0.1)
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self.wan_dir,
                env=my_env
            )
            
            progress_line_count = 0
            progress_total = num_inference_steps
            
            for line in iter(process.stdout.readline, ""):
                logger.debug(line.strip())
                
                if "it]" in line or "it/s]" in line:
                    progress_line_count += 1
                    if progress_callback is not None:
                        progress = 0.1 + (progress_line_count / progress_total) * 0.8
                        progress_callback(min(0.9, progress))
            
            returncode = process.wait()
            
            if returncode != 0:
                error_output = process.stderr.read()
                logger.error(f"WAN generation failed with code {returncode}: {error_output}")
                return None, f"WAN generation failed with code {returncode}: {error_output}"
            
            if not os.path.exists(output_filename):
                logger.error(f"Output video file not found at: {output_filename}")
                return None, "No video file was generated"
            
            video_path = output_filename
            
            import imageio.v2 as imageio
            import numpy as np
            
            reader = imageio.get_reader(video_path)
            frames = []
            for frame in reader:
                frames.append(frame)
            reader.close()
            
            video_tensor = torch.from_numpy(np.array(frames))

            if video_tensor.ndim == 4 and video_tensor.shape[-1] == 3:
                video_tensor = video_tensor.permute(0, 3, 1, 2)
            
            logger.info(f"Video generated successfully with shape: {video_tensor.shape}")
            
            if progress_callback is not None:
                progress_callback(1.0)
            
            return video_tensor, "Video generated successfully with WAN model"
        
        except Exception as e:
            error_msg = f"Error generating video with WAN model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None, error_msg
