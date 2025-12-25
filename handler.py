import base64
import io
import requests
from typing import Any, Dict, List
from PIL import Image

# 모듈 임포트
from eye_analysis_module import analyze_image, generate_report
from eye_rag_chatbot2 import EyeRAGChatbot2, DogEyeCase
import runpod

def _load_image(image_input: str) -> Image.Image:
    if image_input.startswith("http"):
        resp = requests.get(image_input, timeout=10)
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB")

def _format_chat_history(history_list: List[Dict[str, str]]) -> str:
    """API로 받은 히스토리 리스트 -> 프롬프트용 문자열 변환"""
    formatted = ""
    for msg in history_list:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user":
            formatted += f"User: {content}\n"
        elif role in ["assistant", "ai"]:
            formatted += f"AI: {content}\n"
    return formatted.strip()

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("input") or event or {}
    mode = payload.get("mode", "diag")

    # 1) 진단 모드
    if mode == "diag":
        images = payload.get("images") or []
        if not images:
            return {"error": "images required"}
        try:
            img = _load_image(images[0])
            diag_result = analyze_image(img)
            report_md = generate_report(diag_result)
            return {
                "output": report_md,
                "diagnosis": diag_result,
                "case_id": payload.get("case_id", "case_001"),
                "mode": "diag",
            }
        except Exception as e:
            return {"error": str(e)}

    # 2) 챗봇 모드
    elif mode == "chat":
        question = payload.get("question")
        if not question:
            return {"error": "question required"}

        # 진단 정보 복원
        diag_block = payload.get("diagnosis") or {}
        if isinstance(diag_block, dict):
            diagnosis_name = diag_block.get("diagnosis", "").strip()
            symptoms = diag_block.get("symptoms", [])
        else:
            diagnosis_name = str(diag_block)
            symptoms = []

        # 채팅 히스토리 포맷팅
        raw_history = payload.get("chat_history") or []
        history_str = _format_chat_history(raw_history)

        try:
            # 챗봇 인스턴스 생성 (전역 모델 사용)
            bot = EyeRAGChatbot2()
            
            case = bot.start_case(
                case_id=payload.get("case_id", "chat_session"),
                diagnosis=diagnosis_name,
                report_text=payload.get("report_text", ""),
                symptoms=[str(s) for s in symptoms]
            )

            # 답변 생성 (히스토리 전달)
            answer = bot.answer(case, question, history_str)

            return {
                "output": answer,
                "case_id": case.case_id,
                "mode": "chat"
            }
        except Exception as e:
            return {"error": f"Chatbot Error: {str(e)}"}

    return {"error": f"unknown mode: {mode}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})