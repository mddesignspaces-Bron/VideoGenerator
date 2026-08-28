import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# 1. Configuration & Prompt
prompt = "8-bit pixel art neon cyberpunk city street in the rain, puddles reflecting neon lights, atmospheric"
output_filename = "local_test_output.mp4"
model_id = "damo-vilab/text-to-video-ms-1.7b"

# 2. Hardware Detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running generation using device: {device}")

# 3. Load Model Pipeline
print("Loading model weights into memory...")
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None
)

if device == "cuda":
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()  # Optimizes VRAM usage
else:
    print("Warning: Running on CPU will be significantly slower.")

# 4. Generate Video Frames
print(f"Generating video for prompt: '{prompt}'")
result = pipe(
    prompt=prompt,
    negative_prompt="blurry, distorted, low quality, artifacts",
    num_inference_steps=30,
    num_frames=16
)

# 5. Export to MP4
video_frames = result.frames[0]
export_to_video(video_frames, output_filename, fps=8)
print(f"Render complete! Video saved to: {output_filename}")
