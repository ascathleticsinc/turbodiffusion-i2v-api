from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
import torch
import tempfile
import logging
from pathlib import Path
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add turbodiffusion to Python path
sys.path.insert(0, '/app')

app = FastAPI(
    title="TurboDiffusion I2V API",
    description="Image-to-Video generation using TurboWan2.2-I2V-A14B-720P model",
    version="1.0.0"
)

# Global variables for model
model_loaded = False
inference_module = None

@app.on_event("startup")
async def load_model():
    """Load the TurboDiffusion model on startup"""
    global model_loaded, inference_module
    try:
        logger.info("Loading TurboDiffusion model...")
        # Import the inference module
        from turbodiffusion.inference import wan2_2_i2v_infer
        inference_module = wan2_2_i2v_infer
        model_loaded = True
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TurboDiffusion I2V API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Generate video from image (POST)",
            "/docs": "API documentation"
        }
    }

@app.post("/generate")
async def generate_video(
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Text prompt describing the video motion"),
    num_steps: int = Form(4, description="Number of sampling steps (1-4)"),
    num_frames: int = Form(81, description="Number of frames to generate"),
    resolution: str = Form("720p", description="Output resolution (480p or 720p)"),
    seed: int = Form(0, description="Random seed for reproducibility"),
    sigma_max: int = Form(200, description="Initial sigma for rCM")
):
    """Generate video from input image and prompt"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image
            image_path = Path(temp_dir) / "input.jpg"
            contents = await image.read()
            with open(image_path, "wb") as f:
                f.write(contents)
            
            # Define output path
            output_path = Path(temp_dir) / "output.mp4"
            
            logger.info(f"Generating video with prompt: {prompt}")
            logger.info(f"Parameters: steps={num_steps}, frames={num_frames}, resolution={resolution}")
            
            # Run inference
            import subprocess
            cmd = [
                "python", "-m", "turbodiffusion.inference.wan2_2_i2v_infer",
                "--model", "Wan2.2-A14B",
                "--low_noise_model_path", "/app/checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
                "--high_noise_model_path", "/app/checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
                "--vae_path", "/app/checkpoints/Wan2.1_VAE.pth",
                "--text_encoder_path", "/app/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                "--resolution", resolution,
                "--image_path", str(image_path),
                "--prompt", prompt,
                "--num_samples", "1",
                "--num_steps", str(num_steps),
                "--num_frames", str(num_frames),
                "--seed", str(seed),
                "--sigma_max", str(sigma_max),
                "--save_path", str(output_path),
                "--quant_linear",
                "--attention_type", "sagesla",
                "--sla_topk", "0.1",
                "--ode"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Generation failed: {result.stderr}")
                raise HTTPException(status_code=500, detail=f"Video generation failed: {result.stderr}")
            
            logger.info("Video generated successfully")
            
            # Read the generated video
            if output_path.exists():
                return FileResponse(
                    str(output_path),
                    media_type="video/mp4",
                    filename="generated_video.mp4"
                )
            else:
                raise HTTPException(status_code=500, detail="Output video not found")
            
    except Exception as e:
        logger.error(f"Error during video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
