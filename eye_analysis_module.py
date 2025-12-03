#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye_analysis_module.py
----------------------
[수정 사항]
1. eye_rag_chatbot2를 import하여 클래스 및 RAG 로직 사용.
2. handler가 연속 호출되어도 모델은 1회만 로딩.
3. [중요] 챗봇 모드에서는 LoRA 어댑터를 비활성화(disable_adapter)하여 기본 모델로 동작.
4. [New] _generate_vl: 입력 토큰 길이를 계산하여 생성된 부분만 디코딩 (split 불필요).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from ultralytics import YOLO
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# eye_rag_chatbot2 모듈 가져오기
import eye_rag_chatbot2  
from eye_rag_chatbot2 import (
    AppConfig,
    ChatbotState,
    DogEyeCase,
    EyeRAGChatbot2,
    build_ctx_block,
    build_rag_context,
    create_chatbot_state,
)

# -------------------------------
# 설정 및 상수
# -------------------------------

CHECKPOINT_DIR = "/workspace/outputs_py/checkpoint-15020"
BASE_VL_MODEL_NAME = "unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit"
YOLO_WEIGHTS = "/workspace/yolo_weights/best.pt"

DIAG_INSTRUCTION_JSON = """[REPORT_DIAGNOSIS_JSON]
[SYSTEM ROLE]
당신은 동물의 안구 이미지를 분석하는 전문 수의 보조 AI입니다.
입력된 이미지를 면밀히 분석하여 [JSON 형식]으로 결과를 반환하십시오.

[지시 사항]
1. 이미지에서 관찰되는 가장 유력한 [진단명]을 도출하시오.
2. 해당 진단을 뒷받침하는 [핵심 증상]을 찾으시오.

[출력 형식 (JSON)]
반드시 아래 키(Key) 구조를 지켜야 하며, 주석이나 마크다운 코드블록(```json) 없이 순수 JSON 텍스트만 출력하시오.
{
    "diagnosis": "…",
    "symptoms": ["…", …]
}
"""

# -------------------------------
# 전역 모델 변수
# -------------------------------
_YOLO_MODEL: Optional[YOLO] = None
_VL_PROCESSOR = None
_VL_MODEL = None


def _ensure_yolo_loaded() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if not os.path.isfile(YOLO_WEIGHTS):
            raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
        print("[System] Loading YOLO model...")
        _YOLO_MODEL = YOLO(YOLO_WEIGHTS)
    return _YOLO_MODEL


def _ensure_vl_lora_loaded() -> Tuple[Any, Any]:
    global _VL_PROCESSOR, _VL_MODEL
    if _VL_MODEL is not None and _VL_PROCESSOR is not None:
        return _VL_PROCESSOR, _VL_MODEL

    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"LoRA checkpoint dir not found: {CHECKPOINT_DIR}")

    print("[System] Loading Qwen3-VL + LoRA model...")
    _VL_PROCESSOR = AutoProcessor.from_pretrained(BASE_VL_MODEL_NAME, trust_remote_code=True)
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        BASE_VL_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # LoRA 모델 로드 (PeftModel)
    _VL_MODEL = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR).eval()
    
    return _VL_PROCESSOR, _VL_MODEL


def _generate_vl(
    messages: List[Dict[str, Any]],
    image: Optional[Image.Image] = None,
    max_new_tokens: int = 512,
    use_lora: bool = True,  # [유지] LoRA 사용 여부 플래그
) -> str:
    """
    Qwen3-VL 추론 함수.
    입력 토큰 길이를 계산하여, 순수하게 새로 생성된 텍스트만 반환합니다.
    """
    processor, model = _ensure_vl_lora_loaded()

    # 1. 프롬프트 구성
    if hasattr(processor, "apply_chat_template"):
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text_prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content')}" for m in messages])

    # 2. 입력 텐서 생성
    if image is not None:
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    else:
        inputs = processor(text=text_prompt, return_tensors="pt")

    device = getattr(model, "device", None)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # [핵심] 3. 입력 토큰 길이 저장 (Slicing을 위함)
    input_ids_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.4, top_p=0.9)

    # 4. 추론 (use_lora 플래그에 따라 분기)
    if use_lora:
        # LoRA 적용 (기본 상태)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
    else:
        # LoRA 끄고 기본 모델로 동작 (챗봇용)
        # print("[System] Running inference with LoRA adapter DISABLED (Base Model mode)")
        with model.disable_adapter():
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)

    # [핵심] 5. 새로 생성된 토큰만 추출 (Input 이후 부분만 자르기)
    # generated_ids shape: [batch_size, total_len]
    new_tokens = generated_ids[:, input_ids_len:]

    # 6. 디코딩
    if hasattr(processor, "batch_decode"):
        out = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    else:
        out = model.config.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    return out.strip()


def crop_eye_region(image: Image.Image) -> Image.Image:
    model = _ensure_yolo_loaded()
    result = model.predict(source=image, save=False, conf=0.6, max_det=1)[0]
    
    if len(result.boxes) == 0:
        return image

    orig_np = result.orig_img
    orig_pil = Image.fromarray(orig_np[:, :, ::-1]) 
    img_w, img_h = orig_pil.size
    
    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy().astype(int)[0]
    
    padding = 60
    p_x1 = max(0, x1 - padding)
    p_y1 = max(0, y1 - padding)
    p_x2 = min(img_w, x2 + padding)
    p_y2 = min(img_h, y2 + padding)

    return orig_pil.crop((p_x1, p_y1, p_x2, p_y2))


# -------------------------------
# [기능 1] 이미지 진단 (LoRA 사용)
# -------------------------------
def analyze_image(image_input: Union[str, Image.Image]) -> Dict[str, Any]:
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    crop = crop_eye_region(img)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DIAG_INSTRUCTION_JSON},
            ],
        }
    ]
    # 진단은 LoRA 필수 -> use_lora=True
    raw_output = _generate_vl(messages, image=crop, max_new_tokens=512, use_lora=True)

    candidates = []
    for m in re.finditer(r"\{[\s\S]*?\}", raw_output):
        try:
            obj = json.loads(m.group(0))
            if "diagnosis" in obj or "symptoms" in obj:
                candidates.append(obj)
        except Exception:
            continue

    if not candidates:
        raise ValueError(f"JSON parsing failed. Raw output:\n{raw_output}")

    data = candidates[-1]
    if "diagnosis" not in data:
        data["diagnosis"] = ""
    if "symptoms" not in data or not isinstance(data["symptoms"], list):
        data["symptoms"] = []
        
    return data


# -------------------------------
# [기능 2] 진단 보고서 생성 (LoRA 사용)
# -------------------------------
def generate_report(diagnosis_result: Dict[str, Any]) -> str:
    diagnosis = (diagnosis_result.get("diagnosis") or "").strip()
    symptoms_list = diagnosis_result.get("symptoms") or []
    if not isinstance(symptoms_list, list):
        symptoms_list = [str(symptoms_list)]
    symptoms_str = ", ".join(map(str, symptoms_list)) if symptoms_list else "증상 정보 없음"

    cfg = AppConfig.from_env()
    state = create_chatbot_state(cfg)

    case = DogEyeCase(
        case_id="report_case", 
        diagnosis=diagnosis or "미상",
        report_text="",
        image_path=None,
        symptoms=list(map(str, symptoms_list)),
    )

    question_for_rag = (
        f"{diagnosis} 반려견 질환에 대해, 주어진 증상({symptoms_str})을 포함하여 "
        "질병 설명 / 진단 근거 / 주요 발생 원인 / 관리 방법을 보호자에게 설명하기 위한 "
        "핵심 근거 자료를 찾아주세요."
    )

    local_docs = state.local_by_diag.get(case.diagnosis, [])
    ctx_docs = build_rag_context(
        case=case,
        question=question_for_rag,
        local_docs=local_docs,
        config=state.config,
    )
    ctx_numbered = build_ctx_block(ctx_docs)

    report_prompt = f"""[REPORT_EXPLAIN]
당신은 반려동물 안과 보고서를 작성하는 수의사입니다.
아래 [CTX]는 사용자가 검색한 '핵심 근거 자료'입니다.
[CTX]의 내용을 **사실의 근거로 사용하되, 당신의 전문 지식을 더하여** 보호자가 이해하기 쉽게 **자세하고 친절하게 설명**해주세요.

[CTX]
{ctx_numbered}

[보고서 입력]
- 진단명: {diagnosis or '미상'}
- 증상: {symptoms_str}

[작성 요구사항]
- 섹션 순서와 이름은 반드시 다음을 따르세요: 질병에 대한 설명, 진단 근거, 주요 발생 원인, 관리 방법
- 문서 언어: 한국어
- 독자: 초보 보호자
- **말투: '...에요', '...해요' 스타일**
- 중요 단어 **굵은 글씨**
- 마크다운 작성

지금부터 위 요구사항을 엄격히 지켜 보고서를 작성하세요.
"""

    messages = [
        {"role": "system", "content": "당신은 반려동물 안과 보고서를 작성하는 수의사입니다."},
        {"role": "user", "content": report_prompt},
    ]

    # 보고서 작성도 전문적 내용이 필요하므로 LoRA 사용 -> use_lora=True
    report_text = _generate_vl(messages, image=None, max_new_tokens=1024, use_lora=True)
    return report_text


# -------------------------------
# [기능 3] 챗봇 세션 생성 (수정됨: LoRA 끄기 적용)
# -------------------------------

def _patched_call_qwen_local(messages, config=None):
    """
    챗봇 모드에서 호출됩니다.
    [중요] use_lora=False를 전달하여 LoRA 어댑터를 끄고 '기본 모델'로 동작하게 합니다.
    """
    return _generate_vl(messages, image=None, max_new_tokens=512, use_lora=False)

def create_chat_session(
    diagnosis_result: Dict[str, Any], 
    report_text: str,
    case_id: str = "chat_session_1"
) -> Tuple[EyeRAGChatbot2, DogEyeCase]:
    
    # [Monkey Patching]
    # eye_rag_chatbot2의 추론 함수를 우리의 _patched_call_qwen_local로 교체
    if hasattr(eye_rag_chatbot2, "_call_qwen_local"):
        eye_rag_chatbot2._call_qwen_local = _patched_call_qwen_local
    else:
        print("[Warning] Could not patch '_call_qwen_local'. Check module structure.")

    diagnosis = (diagnosis_result.get("diagnosis") or "").strip()
    symptoms_list = diagnosis_result.get("symptoms") or []
    
    bot = EyeRAGChatbot2()
    
    case = bot.start_case(
        case_id=case_id,
        diagnosis=diagnosis or "미상",
        report_text=report_text,
        image_path=None,
        symptoms=list(map(str, symptoms_list)),
    )
    
    return bot, case