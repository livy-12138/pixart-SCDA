#!/usr/bin/env python
from __future__ import annotations
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import random
import gradio as gr
import numpy as np
import uuid
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtAlphaPipeline
from PIL import Image
import torch
from typing import Tuple
from datetime import datetime
from diffusion.sa_solver_diffusers import SASolverScheduler
from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion import DPMS
#from diffusion.data.datasets import ASPECT_RATIO_256_TEST
#from diffusion.data.datasets.utils import ASPECT_RATIO_256_TEST
# 直接定义，避免导入 diffusion.data 模块（会触发 mmcv 错误）
ASPECT_RATIO_256_TEST = {
     '0.25': [128.0, 512.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '3.0': [432.0, 144.0],
     '4.0': [512.0, 128.0]
}


DESCRIPTION = """# PixArt-Alpha 512 Local
        使用本地 `PixArt-XL-2-512x512` Diffusers 权重的离线文本生成图像演示。支持英文提示词、DPM-Solver 和 SA-Solver。
        """
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU. Generation is supported but will be very slow.</p>"

ENABLE_CPU_OFFLOAD = True  # 如果显存足够设为 False，否则设为 True
MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "512"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
PORT = int(os.getenv("DEMO_PORT", "15432"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "1") == "1"
GRADIO_DEBUG = os.getenv("GRADIO_DEBUG", "0") == "1"
WEBP_QUALITY = int(os.getenv("WEBP_QUALITY", "85"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dtype = torch.float16 if device.type == "cuda" else torch.float32

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]


styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
SCHEDULE_NAME = ["DPM-Solver", "SA-Solver"]
DEFAULT_SCHEDULE_NAME = "DPM-Solver"
NUM_IMAGES_PER_PROMPT = 1

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative


project_root = Path(__file__).parent.parent
local_model_path = project_root / "models"

print("Loading T5 Embedder from local models...")
t5_model_dir = project_root / "models" / "PixArt-XL-2-512x512" / "text_encoder"
t5_tokenizer_dir = project_root / "models" / "PixArt-XL-2-512x512" / "tokenizer"

t5 = T5Embedder(
    device=device,
    dir_or_name=str(t5_model_dir),
    local_files_only=True,
    tokenizer_path=str(t5_tokenizer_dir),
    torch_dtype=model_dtype,
    model_max_length=120
)

print("Loading VAE from local model files...")
vae_path = project_root / "models" / "PixArt-XL-2-512x512" / "vae"
vae = AutoencoderKL.from_pretrained(
    str(vae_path),
    torch_dtype=model_dtype,
    local_files_only=True
).to(device)

print("Loading PixArt model from Diffusers format...")
pixart_model_path = project_root / "models" / "PixArt-XL-2-512x512"
pipe = PixArtAlphaPipeline.from_pretrained(
    str(pixart_model_path),
    torch_dtype=model_dtype,
    local_files_only=True,
    use_safetensors=True,
)

if ENABLE_CPU_OFFLOAD and device.type == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe.to(device)

model = pipe.transformer
model.eval()

print("All models loaded successfully!")


def save_image(img):
    unique_name = f'{str(uuid.uuid4())}.webp'
    save_path = os.path.join(f'output/online_demo_img/{datetime.now().date()}')
    os.makedirs(save_path, exist_ok=True)
    unique_name = os.path.join(save_path, unique_name)
    img.save(unique_name, format="WEBP", quality=WEBP_QUALITY, method=4)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def generate(
        prompt: str,
        negative_prompt: str = "",
        style: str = DEFAULT_STYLE_NAME,
        use_negative_prompt: bool = False,
        seed: int = 0,
        width: int = 512,
        height: int = 512,
        schedule: str = 'DPM-Solver',
        dpms_guidance_scale: float = 4.5,
        sas_guidance_scale: float = 3,
        dpms_inference_steps: int = 20,
        sas_inference_steps: int = 25,
        randomize_seed: bool = False,
        use_resolution_binning: bool = True,
        #progress=gr.Progress(track_tqdm=True),
        progress=gr.Progress(),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)
    
    if schedule == 'DPM-Solver':
        if not isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_inference_steps = dpms_inference_steps
        guidance_scale = dpms_guidance_scale
    elif schedule == 'SA-Solver':
        if not isinstance(pipe.scheduler, SASolverScheduler):
            pipe.scheduler = SASolverScheduler.from_config(
                pipe.scheduler.config, algorithm_type='data_prediction',
                tau_func=lambda t: 1 if 200 <= t <= 800 else 0,
                predictor_order=2, corrector_order=2,
            )
        num_inference_steps = sas_inference_steps
        guidance_scale = sas_guidance_scale
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an English prompt.")
    if not use_negative_prompt:
        negative_prompt = ""
    
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt or "")
    
    with torch.inference_mode():
        prompt_embeds, prompt_mask = t5.get_text_embeddings([prompt])
        neg_embeds, neg_mask = t5.get_text_embeddings([negative_prompt])
        prompt_embeds = prompt_embeds.to(dtype=pipe.transformer.dtype)
        neg_embeds = neg_embeds.to(dtype=pipe.transformer.dtype)
        prompt_mask = prompt_mask.to(dtype=torch.long)
        neg_mask = neg_mask.to(dtype=torch.long)
        
        image = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            negative_prompt=None,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_attention_mask=neg_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            use_resolution_binning=use_resolution_binning,
        ).images


    image_paths = [save_image(img) for img in image]
    print(image_paths)
    return image_paths, seed


all_examples = [
    "A small cactus with a happy face in the Sahara desert.",
    "an astronaut sitting in a diner, eating fries, cinematic, analog film",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "stars, water, brilliantly, gorgeous large scale scene, a little girl, in the style of dreamy realism, light gold and amber, blue and pink, brilliantly illuminated in the background.",
    "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
    "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
    "Spectacular Tiny World in the Transparent Jar On the Table, interior of the Great Hall, Elaborate, Carved Architecture, Anatomy, Symetrical, Geometric and Parameteric Details, Precision Flat line Details, Pattern, Dark fantasy, Dark errie mood and ineffably mysterious mood, Technical design, Intricate Ultra Detail, Ornate Detail, Stylized and Futuristic and Biomorphic Details, Architectural Concept, Low contrast Details, Cinematic Lighting, 8k, by moebius, Fullshot, Epic, Fullshot, Octane render, Unreal ,Photorealistic, Hyperrealism",
    "anthropomorphic profile of the white snow owl Crystal priestess , art deco painting, pretty and expressive eyes, ornate costume, mythical, ethereal, intricate, elaborate, hyperrealism, hyper detailed, 3D, 8K, Ultra Realistic, high octane, ultra resolution, amazing detail, perfection, In frame, photorealistic, cinematic lighting, visual clarity, shading , Lumen Reflections, Super-Resolution, gigapixel, color grading, retouch, enhanced, PBR, Blender, V-ray, Procreate, zBrush, Unreal Engine 5, cinematic, volumetric, dramatic, neon lighting, wide angle lens ,no digital painting blur",
    "The parametric hotel lobby is a sleek and modern space with plenty of natural light. The lobby is spacious and open with a variety of seating options. The front desk is a sleek white counter with a parametric design. The walls are a light blue color with parametric patterns. The floor is a light wood color with a parametric design. There are plenty of plants and flowers throughout the space. The overall effect is a calm and relaxing space. occlusion, moody, sunset, concept art, octane rendering, 8k, highly detailed, concept art, highly detailed, beautiful scenery, cinematic, beautiful light, hyperreal, octane render, hdr, long exposure, 8K, realistic, fog, moody, fire and explosions, smoke, 50mm f2.8",
]
NUM_CACHED_EXAMPLES = 9  # 这个变量可以指定缓存的示例数量
cached_examples = all_examples[:NUM_CACHED_EXAMPLES]

with gr.Blocks(css="app/style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Generated images", columns=NUM_IMAGES_PER_PROMPT, show_label=False, format="webp")
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False, visible=True)
        schedule = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=SCHEDULE_NAME,
            value=DEFAULT_SCHEDULE_NAME,
            label="Sampler Schedule",
            visible=True,
        )
        style_selection = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            label="Image Style",
        )
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            visible=True,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
        with gr.Row():
            dpms_guidance_scale = gr.Slider(
                label="DPM-Solver Guidance scale",
                minimum=1,
                maximum=10,
                step=0.1,
                value=4.5,
            )
            dpms_inference_steps = gr.Slider(
                label="DPM-Solver inference steps",
                minimum=5,
                maximum=40,
                step=1,
                value=14,
            )
        with gr.Row():
            sas_guidance_scale = gr.Slider(
                label="SA-Solver Guidance scale",
                minimum=1,
                maximum=10,
                step=0.1,
                value=3,
            )
            sas_inference_steps = gr.Slider(
                label="SA-Solver inference steps",
                minimum=10,
                maximum=40,
                step=1,
                value=25,
            )

    gr.Examples(
        examples=cached_examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            seed,
            width,
            height,
            schedule,
            dpms_guidance_scale,
            sas_guidance_scale,
            dpms_inference_steps,
            sas_inference_steps,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=PORT, share=GRADIO_SHARE, debug=GRADIO_DEBUG
    )
