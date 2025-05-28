import gradio as gr
import traceback
import logging
import utils
import sys
import os

from wan_model import WANModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("videogen.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("videogen-webui")

MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
wan_model = WANModel(cache_dir=MODELS_CACHE_DIR)

def load_model(resolution="480p"):
    """Load the WAN model with specified resolution"""
    if resolution not in ["480p", "720p"]:
        return "Invalid resolution. Choose either 480p or 720p."
    
    if wan_model.device == "cpu":
        gr.Warning("Running on CPU will be extremely slow!")
    
    return wan_model.load_model(resolution)

def generate_video(
    prompt,
    resolution="480p",
    enhance_prompt=True,
    guidance_scale=7.5,
    num_inference_steps=25,
    motion_bucket_id=127,
    fps=16,
    progress=gr.Progress()
):
    """Generate video based on text prompt using the WAN model"""
    if enhance_prompt:
        prompt = utils.enhance_prompt(prompt, strength="medium")
    
    try:
        logger.info(f"Setting model resolution to {resolution}")
        wan_model.load_model(resolution)
        
        width, height = wan_model.get_dimensions()
        
        logger.info(f"Starting video generation with prompt: {prompt}")
        progress(0.1, desc="Starting generation...")
        
        video_frames, status_message = wan_model.generate(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            motion_bucket_id=motion_bucket_id,
            progress_callback=lambda p: progress(0.1 + p * 0.8)
        )
        
        if video_frames is None:
            logger.error(f"Video generation failed: {status_message}")
            return None, status_message
        
        logger.info(f"App received video_frames with shape: {video_frames.shape if hasattr(video_frames, 'shape') else 'unknown'}")
        
        progress(0.9, desc="Converting to video...")
        video_path, conversion_status = utils.frames_to_video(video_frames, fps=fps)
        
        if video_path and os.path.exists(video_path):
            logger.info(f"Video saved to {video_path} ({conversion_status})")
        else:
            logger.error(f"Video path does not exist: {video_path} ({conversion_status})")
            return None, f"Error: Video file not created properly. {conversion_status}"
    except Exception as e:
        logger.error(f"Error in generate_video: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Error generating video: {str(e)}"
    
    metadata = utils.create_video_info_metadata(
        prompt=prompt,
        model_type="WAN 2.1",
        resolution=wan_model.resolution,
        params={
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "motion_bucket_id": motion_bucket_id,
            "fps": fps
        }
    )
    
    progress(1.0, desc="Done!")
    return video_path, f"Generated video with WAN 2.1 model at {fps} FPS"

def build_ui():
    """Build the Gradio interface"""
    with gr.Blocks(title="WAN Video Generation UI") as demo:
        gr.Markdown("# WAN Video Generation UI")
        
        with gr.Tabs():
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### Model Settings")
                            resolution = gr.Radio(
                                ["480p", "720p"], 
                                label="Resolution", 
                                value="480p",
                                interactive=True
                            )
                            model_status = gr.Textbox(label="Model Status", interactive=False)

                        with gr.Group():
                            gr.Markdown("### Prompt")
                            prompt = gr.Textbox(
                                label="Prompt", 
                                placeholder="Describe the video you want to generate...",
                                lines=3
                            )
                            enhance_prompt = gr.Checkbox(
                                label="Enhance Prompt", 
                                value=True,
                                info="Add quality boosting terms to your prompt"
                            )

                        with gr.Group():
                            gr.Markdown("### Generation Parameters")
                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    label="Guidance Scale", 
                                    minimum=1.0, 
                                    maximum=15.0, 
                                    value=7.5, 
                                    step=0.1,
                                    info="How closely to follow the prompt (higher = more faithful but less creative)"
                                )
                            with gr.Row():
                                num_inference_steps = gr.Slider(
                                    label="Inference Steps", 
                                    minimum=10, 
                                    maximum=50, 
                                    value=25, 
                                    step=1,
                                    info="Number of steps for diffusion process. More steps = better quality but slower"
                                )
                                
                                motion_bucket_id = gr.Slider(
                                    label="Motion Intensity", 
                                    minimum=1, 
                                    maximum=255, 
                                    value=127, 
                                    step=1,
                                    info="Controls the amount of motion in the video. Higher = more motion"
                                )
                                fps = gr.Slider(
                                    label="FPS", 
                                    minimum=8, 
                                    maximum=30, 
                                    value=16, 
                                    step=1,
                                    info="WAN recommended default: 16"
                                )
                        
                        with gr.Row():
                            generate_btn = gr.Button("Generate Video", variant="primary", scale=1)
                        
                        with gr.Row():
                            load_button = gr.Button("Load Model", scale=1)
                        
                    with gr.Column(scale=3):
                        with gr.Group():
                            gr.Markdown("### Generated Video")
                            video_output = gr.Video(label="Output Video")
                            generation_info = gr.Textbox(label="Generation Info", interactive=False)
            
            with gr.Tab("History"):
                gr.Markdown("## History Tab (Coming Soon)")
                gr.Markdown("The history tab will allow you to view and manage previously generated videos.")
            
            with gr.Tab("Settings"):
                gr.Markdown("## Settings Tab (Coming Soon)")
                gr.Markdown("The settings tab will include additional configuration options for model loading and generation.")
        
        load_button.click(
            fn=load_model,
            inputs=[resolution],
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                resolution,
                enhance_prompt,
                guidance_scale,
                num_inference_steps,
                motion_bucket_id,
                fps
            ],
            outputs=[
                video_output,
                generation_info
            ]
        )
    
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
