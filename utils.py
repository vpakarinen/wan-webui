import numpy as np
import traceback
import tempfile
import logging
import torch
import os

from PIL import Image

logger = logging.getLogger("videogen-webui")

def frames_to_video(frames, fps=16, output_format="mp4"):
    """Convert a sequence of frames to a video file"""
    try:
        logger.info(f"frames_to_video input type: {type(frames)}, shape: {frames.shape if hasattr(frames, 'shape') else 'unknown'}")
        
        if isinstance(frames, torch.Tensor):
            logger.info("Converting torch tensor to PIL images")
            frames_np = frames.cpu().numpy()
            
            if frames_np.ndim == 5:
                logger.info("Detected 5D tensor, removing batch dimension")
                frames_np = frames_np.squeeze(0)
                
            if frames_np.shape[1] == 3:
                logger.info("Transposing [frames, channels, height, width] to [frames, height, width, channels]")
                frames_np = frames_np.transpose(0, 2, 3, 1)
            
            logger.info(f"Normalizing to 0-255 range, current min: {frames_np.min()}, max: {frames_np.max()}")
            
            if frames_np.min() >= -1 and frames_np.max() <= 1:
                frames_np = ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            elif frames_np.min() >= 0 and frames_np.max() <= 1:
                frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
            else:
                frames_np = frames_np.clip(0, 255).astype(np.uint8)
                
            logger.info(f"Converting {len(frames_np)} numpy arrays to PIL images")
            frames = [Image.fromarray(frame) for frame in frames_np]
            
        elif isinstance(frames, np.ndarray):
            logger.info(f"Converting numpy array with shape {frames.shape} to PIL images")
            if frames.ndim == 4:
                frames = [Image.fromarray(frame) for frame in frames]
            elif frames.ndim == 3:
                frames = [Image.fromarray(frames)]
            else:
                logger.error(f"Unsupported numpy array shape: {frames.shape}")
                raise ValueError(f"Unsupported numpy array shape: {frames.shape}")
                
        elif isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], np.ndarray):
            logger.info(f"Converting list of {len(frames)} numpy arrays to PIL images")
            frames = [Image.fromarray(frame) for frame in frames]
            
        if not frames or len(frames) == 0:
            logger.error("No frames to convert to video")
            return None, "Error: No frames to convert to video"
            
        logger.info(f"Creating temporary directory for video output")
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Temporary directory: {temp_dir}")
        logger.info(f"Saving {len(frames)} frames as GIF with {fps} FPS")
        gif_path = os.path.join(temp_dir, "temp.gif")
        
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,
                loop=0
            )
            logger.info(f"GIF saved to {gif_path}")
        except Exception as e:
            logger.error(f"Error saving GIF: {str(e)}")
            logger.error(traceback.format_exc())
            try:
                logger.info("Trying alternative approach - saving individual frames")
                frame_paths = []
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    frame.save(frame_path)
                    frame_paths.append(frame_path)
                
                gif_path = os.path.join(temp_dir, "temp.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=1000//fps,
                    loop=0
                )
                logger.info(f"GIF created from individual frames and saved to {gif_path}")
            except Exception as inner_e:
                logger.error(f"Error in alternative approach: {str(inner_e)}")
                logger.error(traceback.format_exc())
                raise
        
        if output_format.lower() == "gif":
            logger.info("Returning GIF path as requested")
            return gif_path, "Generated GIF successfully"
        
        try:
            import subprocess
            logger.info("Converting GIF to MP4 using ffmpeg")
            output_path = os.path.join(temp_dir, "output.mp4")
            cmd = [
                "ffmpeg", "-i", gif_path, 
                "-movflags", "faststart",
                "-pix_fmt", "yuv420p", 
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                output_path
            ]
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"MP4 saved to {output_path} with size {os.path.getsize(output_path)} bytes")
                return output_path, "Generated MP4 successfully"
            else:
                logger.error(f"ffmpeg did not produce a valid output file. stdout: {result.stdout}, stderr: {result.stderr}")
                return gif_path, "Could not convert to MP4 (using GIF instead): ffmpeg did not produce output"
                
        except Exception as e:
            logger.error(f"Error converting to MP4: {str(e)}")
            logger.error(traceback.format_exc())
            return gif_path, f"Could not convert to MP4 (using GIF instead): {str(e)}"
            
    except Exception as e:
        logger.error(f"Error in frames_to_video: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Error creating video: {str(e)}"

def enhance_prompt(prompt, strength="medium"):
    """Enhance a prompt with quality boosting terms"""
    if not prompt.rstrip().endswith((".", "!", "?")):
        prompt = prompt.rstrip() + "."

    enhancements = {
        "light": " High quality, detailed.",
        "medium": " High quality, detailed, best quality, extremely detailed.",
        "strong": " High quality, detailed, best quality, extremely detailed, professional, 4K, cinematic lighting, masterpiece."
    }
    
    return prompt + enhancements.get(strength, enhancements["medium"])

def create_video_info_metadata(prompt, model_type, resolution, params):
    """Create metadata for the generated video"""
    return {
        "prompt": prompt,
        "model": model_type,
        "resolution": resolution,
        "parameters": params
    }
