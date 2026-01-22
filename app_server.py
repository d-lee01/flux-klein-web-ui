import os
import sys
import io
import base64
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from pathlib import Path
from contextlib import asynccontextmanager

# Add flux2 src to path if needed (though installed in editable mode)
sys.path.append(os.path.join(os.path.dirname(__file__), "flux2", "src"))

from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder
from flux2.sampling import get_schedule, denoise, denoise_cfg, batched_prc_txt, batched_prc_img, scatter_ids
from einops import rearrange

# --- Configuration ---
# Default to the smallest/fastest model for interactive use
DEFAULT_MODEL = "flux.2-klein-4b"

class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    num_steps: int = 4  # Klein defaults to 4
    guidance: float = 1.0  # Klein defaults to 1.0
    seed: Optional[int] = None
    model_name: str = DEFAULT_MODEL

class ModelManager:
    def __init__(self):
        self.model_name = None
        self.device = self._get_device()
        self.dtype = torch.bfloat16
        self.model = None
        self.ae = None
        self.text_encoder = None
        self.mod_and_upsampling_model = None
        self.model_info = None

    def _get_device(self):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("Using MPS (Metal Performance Shaders) acceleration for Mac.")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA acceleration.")
            return torch.device("cuda")
        else:
            print("WARNING: Using CPU. This will be slow.")
            return torch.device("cpu")

    def load_model(self, model_name: str):
        if self.model_name == model_name:
            return # Already loaded

        print(f"Loading model: {model_name} on {self.device}...")
        
        # Ensure model name exists
        if model_name not in FLUX2_MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_info = FLUX2_MODEL_INFO[model_name]
        
        # Load components
        # Note: We pass device explicitly to ensure everything ends up on MPS/GPU
        self.text_encoder = load_text_encoder(model_name, device=self.device)
        
        if "klein" in model_name:
            # Klein models need a separate text encoder for some reason or reuse?
            # Creating a separate instance as per cli.py logic just in case, 
            # though usually it might be same. cli.py line 287 loads 'flux.2-dev' for upsampling
            # We skip upsampling for basic generation to save memory if not needed,
            # but for compatibility let's follow cli.py if we implement upsampling later.
            # For now, let's keep it simple and just use the text_encoder we have for generation context
            # strict adherence to cli:
            self.mod_and_upsampling_model = load_text_encoder("flux.2-dev", device=self.device)
        else:
            self.mod_and_upsampling_model = self.text_encoder

        self.model = load_flow_model(model_name, device=self.device)
        self.ae = load_ae(model_name, device=self.device)
        
        # Set to eval
        self.model.eval()
        self.ae.eval()
        self.text_encoder.eval()
        self.mod_and_upsampling_model.eval()
        
        self.model_name = model_name
        print(f"Model {model_name} loaded successfully.")

    @torch.inference_mode()
    def generate(self, req: GenerateRequest):
        if self.model_name != req.model_name:
            self.load_model(req.model_name)
            
        print(f"Generating: '{req.prompt}' ({req.width}x{req.height}, steps={req.num_steps})")
        
        # Handle Seed
        if req.seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        else:
            seed = req.seed
        
        # Setup randomness
        # MPS generator handling
        if self.device.type == "mps":
             generator = torch.Generator(device="cpu").manual_seed(seed) # MPS RNG can be finicky, CPU seed for noise generation + to(device) often safer/portable
             # But let's try to follow cli.py: 
             # generator = torch.Generator(device="cuda").manual_seed(seed)
             # adapting for generic device:
             generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
             generator = torch.Generator(device=self.device).manual_seed(seed)

        # 1. Text Encoding
        if self.model_info["guidance_distilled"]:
            ctx = self.text_encoder([req.prompt]).to(self.dtype)
        else:
            ctx_empty = self.text_encoder([""]).to(self.dtype)
            ctx_prompt = self.text_encoder([req.prompt]).to(self.dtype)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
            
        ctx, ctx_ids = batched_prc_txt(ctx) # Moves to device inside? No, expects device.
        # Ensure ctx is on device
        ctx = ctx.to(self.device)
        ctx_ids = ctx_ids.to(self.device)


        # 2. Prepare Noise
        # Shape: (batch, channels, h, w) - latent space is 1/16th of pixel space? 
        # cli.py: shape = (1, 128, height // 16, width // 16) - wait, 128 channels?
        # Yes, flux uses 16 patch size * 16? No, usually 64 or 128 dim for latents.
        # Let's trust cli.py: shape = (1, 128, height // 16, width // 16)
        
        shape = (1, 128, req.height // 16, req.width // 16)
        randn = torch.randn(shape, generator=generator, dtype=self.dtype, device=self.device)
        x, x_ids = batched_prc_img(randn)
        x = x.to(self.device)
        x_ids = x_ids.to(self.device)

        # 3. Denoising Loop
        timesteps = get_schedule(req.num_steps, x.shape[1], shift=True) # Check if shift arg exists in this version of flux2. 
        # flux2/sampling.py might have different signature. 
        # checking cli.py line 584: timesteps = get_schedule(cfg.num_steps, x.shape[1]) -> No shift arg.
        timesteps = get_schedule(req.num_steps, x.shape[1])

        # img_cond_seq for I2I is None for T2I
        img_cond_seq = None
        img_cond_seq_ids = None

        if self.model_info["guidance_distilled"]:
            x = denoise(
                self.model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=req.guidance
            )
        else:
            x = denoise_cfg(
                self.model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=req.guidance
            )

        # 4. Decode
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        print("Decoding latents...")
        x = self.ae.decode(x).float()
        
        # 5. Post-process
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        
        # Convert to Base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image_base64": img_str, "seed": seed}

# Global Model Manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load default model
    try:
        model_manager.load_model(DEFAULT_MODEL)
    except Exception as e:
        print(f"Error loading default model: {e}")
    yield
    # Shutdown: Clean up if needed
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.post("/generate")
def generate_image(req: GenerateRequest):
    try:
        return model_manager.generate(req)
    except Exception as e:
        print(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    return {"models": list(FLUX2_MODEL_INFO.keys()), "current": model_manager.model_name}

# Serve Static Files
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
