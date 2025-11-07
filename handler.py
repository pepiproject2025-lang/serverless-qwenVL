import os, io, base64, json
from typing import List, Dict, Any
from PIL import Image

# Unsloth FastVisionModel : Qwen2.5-VL에 맞는 헬퍼
from unsloth import FastVisionModel

import torch
import runpod
from transformers import AutoProcessor
from peft import PeftModel

# ===== 환경 변수 =====
# Runpod 콘솔에서 Volumes : <네트워크 볼륨> -> Mount Path: /runpod-volume
# worker -> runpod-volume으로 수정
MODEL_BASE = os.getenv("MODEL_BASE", "/runpod-volume/models/Qwen2.5_VL_7B_Instruct")
LORA_PATH = os.getenv("LORA_PATH", "/runpod-volume/outputs/checkpoint-11200")
LOAD_4BIT = os.getenv("LOAD_4BIT", "true").lower() == "true"
MERGE_LORA = os.getenv("MERGE_LORA", "false").lower() == "true"     # 병합 고정 옵션

# ===== 모델 로드 =====
print(f"[INIT] MODEL_BASE={MODEL_BASE}")
print(f"[INIT] LORA_PATH={LORA_PATH}")
print(f"[INIT] LOAD_4BIT={LOAD_4BIT}, MERGE_LORA={MERGE_LORA}")

torch.set_grad_enabled(False)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = MODEL_BASE,
    load_in_4bit = LOAD_4BIT,
)

# LoRA 적용 (가능한 경로들을 안전하게 시도)
def _apply_lora_if_available(base_model):
    if not LORA_PATH or not os.path.exists(LORA_PATH):
        print("[LORA] path missing or not found, skip.")
        return base_model
    
    print(f"[LORA] Trying to load LoRA from: {LORA_PATH}")
    # 1) 우선 PeftModel 경로 (일반적)
    try:
        peft_model = PeftModel.from_pretrained(base_model, LORA_PATH)
        peft_model.eval()
        print("[LORA] Loaded via PeftModel.from_pretrained.")
        return peft_model
    except Exception as e:
        print(f"[LORA] Peft load failed: {e}")

    # 2) Unsloth 래퍼에 load_adapter가 있는 경우
    try:
        if hasattr(base_model, "load_adapter"):
            base_model.load_adapter(LORA_PATH)
            print("[LORA] Loaded via base_model.load_adapter.")
            return base_model
    except Exception as e:
        print(f"[LORA] base_model.load_adapter failed: {e}")

    print("[LORA] No LoRA applied.")
    return base_model

model = _apply_lora_if_available(model)

# (선택) 배포 고정/속도 개선: LoRA 병합
# if MERGE_LORA:
#     try:
#         model = model.merge_and_unload()
#         print("[LORA] merge_and_upload done.")
#     except Exception as e:
#         print(f"[LORA] merge failed: {e}")

FastVisionModel.for_inference(model)
processor = AutoProcessor.from_pretrained(MODEL_BASE)

# ===== 유틸 =====
def _normalize_github_url(url: str) -> str:
    # blob → raw, refs/heads → branch 이름 정리, 쿼리 제거
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    url = url.replace("/refs/heads/", "/")
    if "raw.githubusercontent.com" in url and "?" in url:
        url = url.split("?", 1)[0]
    return url

def _load_image(x: str) -> Image.Image:
    if x.startswith(("http://", "https://")):
        import requests
        url = _normalize_github_url(x)
        try:
            r = requests.get(url, timeout=10, allow_redirects=True)
            status = r.status_code
            ctype = r.headers.get("Content-Type", "")
            print(f"[IMAGE] GET {url} -> {status} {ctype}")
            # 이미지가 아니면 본문 일부를 같이 출력
            if not ctype.lower().startswith("image/"):
                snippet = r.text[:200].replace("\n"," ")
                raise RuntimeError(f"Non-image content: status={status}, content-type={ctype}, body[:200]={snippet}")
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Image fetch failed: {url} | {type(e).__name__}: {e}")
    # base64
    try:
        return Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Invalid base64 image: {type(e).__name__}: {e}")


def _infer(prompt: str, images: List[str], gen: Dict[str, Any] | None):
    pil_imgs = [_load_image(s) for s in images]
    inputs = processor(text=prompt, images=pil_imgs, return_tensors="pt").to(model.device)

    gen = gen or {}
    max_new_tokens  = int(gen.get("max_new_tokens", 512))
    temperature      = float(gen.get("temperature", 0.2))
    top_p           = float(gen.get("top_p", 0.9))
    top_k           = int(gen.get("top_k", 50))

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# ===== Runpod 핸들러 =====
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    기대 입력(JSON):
    {
        "input": {
            "prompt": "...",
            "images": ["<url or base64>", "..."],
            "gen": {"max_new_tokens": 512, "temperature":0.2}
        }
    }
    """
    try:
        payload = event.get("input", {}) if isinstance(event, dict) else {}
        prompt  = payload.get("prompt", "Describe the image.")
        images  = payload.get("images", [])
        gen     = payload.get("gen", {})

        if not images:
            return {"error": "images required (url or base64 list)"}
        
        result = _infer(prompt, images, gen)
        return {"output": result}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()[:4000]
        print("[ERROR]", tb)
        return {"error": f"{e}", "trace": tb}
    

if __name__ == "__main__":
    # Runpod serverless runtime
    runpod.serverless.start({"handler": handler})
