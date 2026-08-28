import os
import uuid
import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# Initialize server application
app = FastAPI(title="Local Video Synthesis Engine", version="1.0.0")

# Setup output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model state
pipeline = None

class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    num_inference_steps: int = 35
    num_frames: int = 16
    fps: int = 8

@app.on_event("startup")
def load_model():
    """Loads the model into local GPU memory once on server start."""
    global pipeline
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    
    pipeline = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    
    # Push model to local CUDA GPU with memory optimization
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cpu")

def process_video_generation(task_id: str, request: VideoRequest):
    """Executes local rendering loop."""
    output_filename = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")
    
    # Run the model locally
    result = pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        num_frames=request.num_frames
    )
    
    # Export raw frames to video
    export_to_video(result.frames[0], output_filename, fps=request.fps)

@app.post("/generate")
def create_video_task(request: VideoRequest, background_tasks: BackgroundTasks):
    """API endpoint to receive prompts and queue generation locally."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    task_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(process_video_generation, task_id, request)
    
    return {
        "status": "queued",
        "task_id": task_id,
        "output_path": f"{OUTPUT_DIR}/{task_id}.mp4"
    }

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "gpu_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
