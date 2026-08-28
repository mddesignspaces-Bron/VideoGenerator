import requests

SERVER_URL = "http://localhost:8000/generate"

prompts = [
    "8-bit pixel art neon cyberpunk alleyway in the rain, cinematic",
    "Cozy retro bedroom looking out at a swirling galaxy nebula",
    "Futuristic arcade terminal with glowing screens and CRT scanlines"
]

for item in prompts:
    payload = {
        "prompt": item,
        "num_inference_steps": 30,
        "num_frames": 16,
        "fps": 8
    }
    res = requests.post(SERVER_URL, json=payload)
    print(f"Dispatched: {res.json()}")
