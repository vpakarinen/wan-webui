import numpy as np
import tempfile
import os

from PIL import Image

def frames_to_video(frames, fps=16, output_format="mp4"):
    """Convert a sequence of frames to a video file"""
    try:
        if isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(frame) for frame in frames]

        temp_dir = tempfile.mkdtemp()
        
        gif_path = os.path.join(temp_dir, "temp.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,
            loop=0
        )
        
        if output_format.lower() == "gif":
            return gif_path, "Generated GIF successfully"
        
        try:
            import subprocess
            output_path = os.path.join(temp_dir, "output.mp4")
            cmd = [
                "ffmpeg", "-i", gif_path, 
                "-movflags", "faststart",
                "-pix_fmt", "yuv420p", 
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path, "Generated MP4 successfully"
        except Exception as e:
            return gif_path, f"Could not convert to MP4 (using GIF instead): {str(e)}"
    except Exception as e:
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
