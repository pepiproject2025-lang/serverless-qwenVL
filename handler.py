import base64
import io
from typing import Any, Dict, List

from PIL import Image

# 기존 분석 모듈 (그대로 유지)
from eye_analysis_module import (
    analyze_image,
    generate_report,
)
# 수정된 챗봇 모듈 임포트
from eye_rag_chatbot2 import EyeRAGChatbot2, DogEyeCase

import runpod
import requests

def _load_image(image_input: str) -> Image.Image:
    """
    image_input이 URL이면 URL에서 다운로드,
    base64이면 base64 디코딩으로 처리.
    """
    if image_input.startswith("http://") or image_input.startswith("https://"):
        try:
            resp = requests.get(image_input, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")

    try:
        return _decode_base64_image(image_input)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")


def _decode_base64_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def _format_chat_history(history_list: List[Dict[str, str]]) -> str:
    """
    API로 받은 채팅 히스토리 리스트를 프롬프트용 문자열로 변환
    Input: [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요"}]
    Output: "User: 안녕\nAI: 안녕하세요"
    """
    formatted_hist = ""
    for msg in history_list:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user":
            formatted_hist += f"User: {content}\n"
        elif role == "assistant" or role == "ai":
            formatted_hist += f"AI: {content}\n"
    return formatted_hist.strip()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    [Input]
    {
      "input": {
        "mode": "diag" | "chat",
        
        # diag 모드
        "images": ["<base64>"], 
        "case_id": "user_123",

        # chat 모드
        "case_id": "user_123",
        "question": "집에서 관리법 알려줘",
        "diagnosis": {"diagnosis": "결막염", "symptoms": ["충혈"]},
        "report_text": "...",
        "chat_history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
      }
    }
    """
    payload = event.get("input") or event or {}
    mode = payload.get("mode", "diag")

    # 1) 진단 + 보고서 생성
    if mode == "diag":
        images = payload.get("images") or []
        if not images:
            return {"error": "images[0] (base64) is required for diag mode"}

        try:
            img = _load_image(images[0])
            
            # 1-1. 진단 JSON
            diag_result = analyze_image(img)

            # 1-2. 보고서 마크다운
            report_md = generate_report(diag_result)
            
            req_case_id = payload.get("case_id") or "case_001"

            return {
                "output": report_md,        
                "diagnosis": diag_result,   
                "case_id": req_case_id,    
                "mode": "diag",
            }
        except Exception as e:
            return {"error": str(e)}

    # 2) 챗봇 Q&A
    elif mode == "chat":
        case_id = payload.get("case_id") or "chat_session"
        question = payload.get("question") or payload.get("prompt")
        
        if not question:
            return {"error": "question is required for chat mode"}

        # 2-1. 진단 정보 복원
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

        # 2-2. 채팅 히스토리 처리
        raw_history = payload.get("chat_history") or []
        chat_history_str = _format_chat_history(raw_history)

        # 2-3. 챗봇 초기화 (전역 모델 캐시 사용으로 빠름)
        try:
            bot = EyeRAGChatbot2()
            
            case = bot.start_case(
                case_id=case_id,
                diagnosis=diagnosis_name,
                report_text=report_text,
                symptoms=[str(s) for s in symptoms],
            )

            # 2-4. 답변 생성 (Agent 실행)
            answer = bot.answer(
                case_id=case.case_id, 
                question=question, 
                case=case, 
                chat_history_str=chat_history_str
            )
            
            return {
                "output": answer,
                "case_id": case_id,
                "mode": "chat",
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Chatbot Error: {str(e)}"}

    # 3) 알 수 없는 모드
    else:
        return {"error": f"unknown mode: {mode}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})