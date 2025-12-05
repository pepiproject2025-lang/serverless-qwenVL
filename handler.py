import base64
import io
from typing import Any, Dict

from PIL import Image

from eye_analysis_module import (
    analyze_image,
    generate_report,
    create_chat_session,
)
from eye_rag_chatbot2 import EyeRAGChatbot2, DogEyeCase

import runpod

import os, sys
print("=== DEBUG: sys.path ===")
print(sys.path)
print("=== DEBUG: cwd ===")
print(os.getcwd())
print("=== DEBUG: files in cwd ===")
print(os.listdir("."))
print("==============================")

# 간단한 세션 저장 (case_id -> (bot, case))
# SESSIONS: dict[str, tuple[EyeRAGChatbot2, DogEyeCase]] = {}

import requests

def _load_image(image_input: str) -> Image.Image:
    """
    image_input이 URL이면 URL에서 다운로드,
    base64이면 base64 디코딩으로 처리.
    """
    # URL 형태인지 확인
    if image_input.startswith("http://") or image_input.startswith("https://"):
        try:
            resp = requests.get(image_input, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")

    # URL이 아니면 base64라고 가정
    try:
        return _decode_base64_image(image_input)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")


def _decode_base64_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    기대하는 입력 형식 (예시):

    {
      "input": {
        "mode": "diag" | "chat",
        "images": ["<base64>"],   # diag일 때
        "case_id": "user_123",    # chat일 때
        "question": "집에서 어떻게 관리해요?"  # chat일 때
      }
    }
    """
    payload = event.get("input") or event or {}
    mode = payload.get("mode", "diag")

    # 1) 진단 + 보고서 + 챗봇 세션 초기화
    if mode == "diag":
        images = payload.get("images") or []
        if not images:
            return {"error": "images[0] (base64) is required for diag mode"}

        img = _load_image(images[0])

        # 1-1. 진단 JSON
        diag_result = analyze_image(img)

        # 1-2. 보고서 마크다운
        report_md = generate_report(diag_result)

        # 1-3. 챗봇 세션 생성
        req_case_id = payload.get("case_id") or "case_001"
        #bot, case = create_chat_session(diag_result, report_md, case_id=req_case_id)

        # 실제로 사용할 case_id는 case 객체에서 가져오기
        #real_case_id = case.case_id
        #SESSIONS[real_case_id] = (bot, case)

        # Runpod 응답
        return {
            "output": report_md,        # FastAPI가 그냥 마크다운 보고서만 써도 되고
            "diagnosis": diag_result,   # 추가로 JSON 진단 정보도 같이 넘겨줌
            "case_id": req_case_id,    # 나중에 chat 모드에서 사용할 세션 ID
            "mode": "diag",
        }

    # 2) 챗봇 Q&A
    elif mode == "chat":
        case_id = payload.get("case_id") or "chat_session"
        question = payload.get("question") or payload.get("prompt")
        answer_mode = payload.get("answer_mode", "brief")
        if not question:
            return {"error": "case_id and question are required for chat mode"}

        # if case_id not in SESSIONS:
        #     return {"error": f"case_id '{case_id}' not found (diag 먼저 호출 필요)"}

        # bot, case = SESSIONS[case_id]
        # answer = bot.answer(case.case_id, question, mode=answer_mode)


        diagnosis_block = payload.get("diagnosis") or {}
        if isinstance(diagnosis_block, dict):
            diagnosis_name = (diagnosis_block.get("diagnosis") or "").strip()
            symptoms = diagnosis_block.get("symptoms") or []
        else:
            diagnosis_name = str(diagnosis_block)
            symptoms = []

        report_text = (
            payload.get("report_text")
            or payload.get("report")
            or ""
        )

        bot = EyeRAGChatbot2()

        case = bot.start_case(
            case_id=case_id,
            diagnosis=diagnosis_name,
            report_text=report_text,
            symptoms=[str(s) for s in symptoms],
        )

        answer = bot.answer(case.case_id, question, mode=answer_mode)
        
        return {
            "output": answer,
            "case_id": case_id,
            "mode": "chat",
            "answer_mode": answer_mode,
        }

    # 3) 알 수 없는 모드
    else:
        return {"error": f"unknown mode: {mode}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
