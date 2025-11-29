import os, io, base64, json
from typing import List, Dict, Any
from PIL import Image

# Unsloth FastVisionModel : Qwen2.5-VL에 맞는 헬퍼
from unsloth import FastVisionModel

import torch
import runpod
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ===== 환경 변수 =====
# Runpod 콘솔에서 Volumes : <네트워크 볼륨> -> Mount Path: /runpod-volume
# worker -> runpod-volume으로 수정
MODEL_BASE = os.getenv("MODEL_BASE", "/runpod-volume/models/Qwen3_VL_8B_Instruct")
LORA_PATH = os.getenv("LORA_PATH", "") # 로라 체크포인트 넣기
LOAD_4BIT = os.getenv("LOAD_4BIT", "true").lower() == "true"
MERGE_LORA = os.getenv("MERGE_LORA", "false").lower() == "true"     # 병합 고정 옵션

MODELS_ROOT = "/runpod-volume/models"

if not os.path.exists(MODELS_ROOT):
    print(f"[CHECK] {MODELS_ROOT} 가 존재하지 않습니다.")
else:
    print(f"[CHECK] {MODELS_ROOT} 목록:")
    try:
        for name in os.listdir(MODELS_ROOT):
            print("  -", name)
    except Exception as e:
        print(f"[CHECK] 모델 목록 조회 실패: {e}")

if not os.path.exists(MODEL_BASE):
    print(f"[ERROR] MODEL_BASE 경로가 존재하지 않습니다: {MODEL_BASE}")
else:
    print(f"[CHECK] MODEL_BASE 경로 확인됨: {MODEL_BASE}")

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

# ===== LoRA on/off 제어 =====
HAS_LORA = isinstance(model, PeftModel)

# PeftModel 안에 로드된 어댑터 이름 (대부분 "default" 하나임)
DEFAULT_ADAPTER_NAME = None
if HAS_LORA:
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        DEFAULT_ADAPTER_NAME = list(peft_config.keys())[0]
        print(f"[LORA] available adapter = {DEFAULT_ADAPTER_NAME}")

def set_lora(enabled: bool):
    """
    요청마다 LoRA 사용 여부를 바꾸기 위한 헬퍼.
    PeftModel 의 set_adapter / disable_adapter 를 이용한다.
    """
    if not HAS_LORA:
        return

    try:
        if enabled:
            name = DEFAULT_ADAPTER_NAME or "default"
            if hasattr(model, "set_adapter"):
                model.set_adapter(name)
            print(f"[LORA] enabled (adapter={name})")
        else:
            if hasattr(model, "disable_adapter"):
                model.disable_adapter()
                print("[LORA] disabled via disable_adapter()")
            else:
                # disable_adapter 없는 구버전이면 완전히 끄지 못할 수도 있음
                print("[LORA] disable_adapter() not found, cannot fully disable.")
    except Exception as e:
        print(f"[LORA] set_lora({enabled}) failed: {e}")

# 초기 상태는 '진단 모드' 기준으로 LoRA 켜두기
set_lora(True)

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


def _count_image_tokens(txt: str) -> int:
    return txt.count("<|image_pad|>")

def _infer(prompt: str, images: List[str], gen: Dict[str, Any] | None):
    pil_imgs = [_load_image(s) for s in images]

    # 1) 우선 공식 템플릿 시도
    messages = [{
        "role": "user",
        "content": [
            *([{"type": "image", "image": img} for img in pil_imgs]),
            {"type": "text", "text": prompt},
        ],
    }]

    try:
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        text = None

    # 2) 템플릿 결과 점검: 이미지 토큰이 0개면 수동으로 삽입 (확실한 방법)
    if not text or _count_image_tokens(text) == 0:
        image_tokens = "".join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in pil_imgs])
        # (선택) 대화 포맷 토큰이 필요한 버전 대비
        text = f"<|im_start|>user\n{image_tokens}\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

    # 안전 검사: 이미지 개수 == 토큰 개수
    n_tok = _count_image_tokens(text)
    assert n_tok == len(pil_imgs), f"image tokens {n_tok} != images {len(pil_imgs)}"

    # 3) 입력 만들기 & 생성
    inputs = processor(text=text, images=pil_imgs, return_tensors="pt").to(model.device)

    gen = gen or {}
    output = model.generate(
        **inputs,
        max_new_tokens=int(gen.get("max_new_tokens", 512)),
        temperature=float(gen.get("temperature", 0.2)),
        top_p=float(gen.get("top_p", 0.9)),
        top_k=int(gen.get("top_k", 50)),
    )

    # 🔧 핵심: 입력 길이 이후만 디코드
    input_len = inputs["input_ids"].shape[1]
    gen_only = output[:, input_len:]
    text = tokenizer.decode(gen_only[0], skip_special_tokens=True)
    return text



# ===== Runpod 핸들러 =====
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    기대 입력(JSON):

    {
      "input": {
        "prompt": "질문 또는 진단용 프롬프트",
        "images": ["<url or base64>", ...],    
        "gen": {"max_new_tokens": 512, "temperature":0.2}                 

        # 새로 추가되는 필드
        "mode": "diag" | "chat",         # 진단(diag) 또는 챗(chat)
        "use_lora": true | false         
      }
    }
    """
    try:
        payload = event.get("input") or event or {}
        prompt = payload.get("prompt", "Describe the image.")
        images = payload.get("images") or []
        gen = payload.get("gen") or {}

        # 1) 모드 & LoRA 사용 여부 결정
        mode = payload.get("mode", "diag")  # "diag" or "chat"
        use_lora_flag = payload.get("use_lora")
        if use_lora_flag is None:
            # 명시 안 해주면: 진단(diag)만 LoRA 사용, 챗(chat)은 LoRA 끔
            use_lora_flag = (mode == "diag")

        if HAS_LORA:
            set_lora(bool(use_lora_flag))
            print(f"[HANDLER] mode={mode}, use_lora={use_lora_flag}")
        else:
            print(f"[HANDLER] mode={mode}, HAS_LORA=False")

        # 2) 모드별 기본 검증
        if mode == "diag" and not images:
            # 진단 모드에서는 이미지를 반드시 요구
            return {"error": "images required (url or base64 list) for diag mode"}

        # chat 모드에서는 images 없어도 _infer 가 텍스트만으로 동작함
        result = _infer(prompt, images, gen)

        return {
            "output": result,
            "mode": mode,
            "used_lora": bool(use_lora_flag) and HAS_LORA,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()[:4000]
        print("[ERROR]", tb)
        return {"error": f"{e}", "trace": tb}
    

if __name__ == "__main__":
    # Runpod serverless runtime
    runpod.serverless.start({"handler": handler})
