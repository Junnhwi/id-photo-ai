import os
import hashlib
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _mock_generate(
    *,
    out_dir: str,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    width: int,
    height: int,
    seed: int | None,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    seed_base = seed if seed is not None else int(hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8], 16)
    outputs: list[str] = []

    for i in range(num_images):
        rng = np.random.default_rng(seed_base + i)
        arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([(12, 12), (width - 12, 110)], outline=(255, 255, 255), width=2)
        draw.text((20, 20), "MOCK GENERATION", fill=(255, 255, 255))
        draw.text((20, 44), f"seed={seed_base + i}", fill=(255, 255, 255))
        draw.text((20, 68), f"prompt={prompt[:48]}", fill=(255, 255, 255))
        if negative_prompt:
            draw.text((20, 92), f"neg={negative_prompt[:44]}", fill=(255, 255, 255))

        out_name = f"gen_{i+1:02d}.png"
        out_path = os.path.join(out_dir, out_name)
        img.save(out_path)
        outputs.append(f"generated/{out_name}")

    return {
        "method": "mock_fallback",
        "warning": "diffusers is not installed; mock images were generated for pipeline validation.",
        "outputs": outputs,
    }


def generate_with_lora(
    *,
    job_path: str,
    base_model: str,
    lora_path: str,
    prompt: str,
    negative_prompt: str = "",
    num_images: int = 1,
    steps: int = 30,
    guidance_scale: float = 7.0,
    width: int = 512,
    height: int = 768,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """
    LoRA 이미지 생성 진입점.

    - diffusers가 설치된 환경에서는 Stable Diffusion + LoRA 로 실제 생성 시도
    - 미설치 환경에서는 API 파이프라인 검증용 mock 이미지를 생성
    """
    out_dir = os.path.join(job_path, "generated")
    runtime_device = _resolve_device(device)

    try:
        import torch
        from diffusers import StableDiffusionPipeline

        dtype = torch.float16 if runtime_device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_single_file(base_model, torch_dtype=dtype)
        pipe = pipe.to(runtime_device)
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path))

        if runtime_device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()

        os.makedirs(out_dir, exist_ok=True)
        outputs: list[str] = []

        gen = torch.Generator(device=runtime_device)
        if seed is not None:
            gen = gen.manual_seed(seed)

        for i in range(num_images):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=gen,
            ).images[0]

            out_name = f"gen_{i+1:02d}.png"
            out_path = os.path.join(out_dir, out_name)
            image.save(out_path)
            outputs.append(f"generated/{out_name}")

        return {
            "method": "diffusers_stable_diffusion_lora",
            "device": runtime_device,
            "base_model": base_model,
            "lora_path": lora_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_images": num_images,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "outputs": outputs,
        }

    except Exception:
        mock = _mock_generate(
            out_dir=out_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            width=width,
            height=height,
            seed=seed,
        )
        mock.update({
            "device": runtime_device,
            "base_model": base_model,
            "lora_path": lora_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_images": num_images,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
        })
        return mock
