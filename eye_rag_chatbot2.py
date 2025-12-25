#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye_rag_chatbot2.py
"""

import os
import torch
import re
from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass, field

# Transformers
from transformers import AutoModelForVision2Seq, AutoProcessor

# LangChain Core
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate

# LangChain Community Tools
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# ⭐️ [Pod 환경 표준] langchain_classic 사용
from langchain_classic.agents import create_react_agent, AgentExecutor

# ----------------------------------
# 전역 설정 & 모델 캐싱
# ----------------------------------
MODEL_DIR = "/runpod-volume/models/Qwen3_VL_8B_Instruct"
CORPUS_DIR = "/runpod-volume/corpus/"

_GLOBAL_MODEL = None
_GLOBAL_PROCESSOR = None

def load_global_model():
    global _GLOBAL_MODEL, _GLOBAL_PROCESSOR
    if _GLOBAL_MODEL is not None:
        return _GLOBAL_MODEL, _GLOBAL_PROCESSOR

    print(f"Loading Model from {MODEL_DIR}...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_DIR, device_map="auto", torch_dtype="auto", trust_remote_code=True
        ).eval()
        processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print("Model Loaded Successfully!")
        _GLOBAL_MODEL = model
        _GLOBAL_PROCESSOR = processor
        return model, processor
    except Exception as e:
        print(f"Model Load Failed: {e}")
        raise e

# ----------------------------------
# [필수] 레거시 데이터 클래스 복구
# ----------------------------------
@dataclass
class AppConfig:
    use_ddg: bool = True
    use_wiki: bool = True
    k: int = 12
    wiki_pages: int = 1
    corpus_dir: str = "/runpod-volume/corpus"
    qwen_local_model_dir: str = "/runpod-volume/models/Qwen3_VL_8B_Instruct"
    answer_max_lines: int = 5
    max_history_turns: int = 4

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls()

@dataclass
class DogEyeCase:
    case_id: str
    diagnosis: str
    report_text: str
    symptoms: List[str] = field(default_factory=list)
    image_path: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class ChatbotState:
    config: AppConfig
    local_by_diag: Dict[str, Any] = field(default_factory=dict)
    cases: Dict[str, DogEyeCase] = field(default_factory=dict)

def create_chatbot_state(config: Optional[AppConfig] = None) -> ChatbotState:
    return ChatbotState(config=config or AppConfig())

# ----------------------------------
# [필수] 레거시 함수 복구 (임포트 에러 방지용)
# 실제 로직은 안 쓰이더라도 eye_analysis_module이 import하므로 존재해야 함
# ----------------------------------
def load_local_corpus_by_diag(corpus_dir: str = "./corpus") -> Dict[str, List[Dict[str, Any]]]:
    """레거시 호환용 더미 함수"""
    return {}

def split_report_into_docs(case: DogEyeCase, max_chars: int = 900) -> List[Dict[str, Any]]:
    """레거시 호환용 더미 함수"""
    return []

def ddg_search(queries: List[str], max_results: int = 6) -> Tuple[List[Dict[str, Any]], List[str]]:
    """레거시 호환용 더미 함수"""
    return [], []

def wiki_chunks(query: str, max_pages: int = 1, max_chars: int = 2500) -> List[Dict[str, Any]]:
    """레거시 호환용 더미 함수"""
    return []

def build_rag_context(case: DogEyeCase, question: str, local_docs: List[Dict[str, Any]], config: AppConfig) -> List[Dict[str, Any]]:
    """
    eye_analysis_module.py에서 임포트하는 함수. 
    호환성을 위해 빈 리스트를 반환하거나 기본 구조만 유지합니다.
    """
    return []

def build_ctx_block(ctx_docs: List[Dict[str, Any]]) -> str:
    """단순 텍스트 반환으로 호환성 유지"""
    return ""

def start_case(state: ChatbotState, case_id: str, diagnosis: str, report_text: str, image_path: Optional[str] = None, symptoms: Optional[List[str]] = None) -> DogEyeCase:
    """레거시 호환 래퍼: 내부적으로 새로운 봇 로직 사용 가능하도록 연결"""
    bot = EyeRAGChatbot2()
    case = bot.start_case(case_id, diagnosis, report_text, symptoms, image_path)
    state.cases[case_id] = case
    return case

def answer_question(state: ChatbotState, case_id: str, question: str, mode: str = "brief") -> str:
    """레거시 호환 래퍼: handler.py가 이 함수를 호출할 경우를 대비"""
    case = state.cases.get(case_id)
    if not case: return "케이스 없음"
    
    # 히스토리 변환
    history_str = ""
    for msg in case.history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user": history_str += f"User: {content}\n"
        elif role == "assistant": history_str += f"AI: {content}\n"
    
    bot = EyeRAGChatbot2()
    answer = bot.answer(case, question, history_str)
    
    case.history.append({"role": "user", "content": question})
    case.history.append({"role": "assistant", "content": answer})
    return answer

# ----------------------------------
# [메인] 새로운 LangChain 로직
# ----------------------------------
class QwenVLLLM(LLM):
    model: Any = None
    processor: Any = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=None, videos=None, padding=True, return_tensors="pt").to(self.model.device)
        gen_kwargs = {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.1, "repetition_penalty": 1.1, "top_p": 0.9, **kwargs}
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if stop:
            for s in stop:
                if s in output_text: output_text = output_text.split(s)[0]
        return output_text

    @property
    def _llm_type(self) -> str: return "qwen-vl-custom"

def load_local_knowledge(diagnosis_name: str) -> str:
    filename = f"{diagnosis_name}.txt"
    filepath = os.path.join(CORPUS_DIR, filename)
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f: return f.read()
        return "해당 질환에 대한 내부 상세 지침 파일이 존재하지 않습니다."
    except Exception as e:
        return f"내부 문서 로딩 중 오류 발생: {e}"

class EyeRAGChatbot2:
    def __init__(self, config: Optional[AppConfig] = None):
        model, processor = load_global_model()
        self.llm = QwenVLLLM(model=model, processor=processor)
        wrapper = DuckDuckGoSearchAPIWrapper(backend="html", max_results=5)
        self.search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
        self.tools = [self.search_tool]
        
        self.template = """
당신은 **'친절하고 전문적인 반려동물 안과 수의사 AI'**입니다.
보호자의 걱정에 공감하며, [진단 요약]과 [원내 의학 지침]을 바탕으로 정확하고 이해하기 쉽게 설명해 주세요.

[사용 가능한 도구]
{tools}

[형식 가이드 - 시스템 에러 방지용]
답변 시 아래 두 가지 형식 중 하나를 반드시 선택하세요.

**상황 1. 검색이 필요한 경우:**
Question: (질문)
Thought: 추가 정보를 검색해야 합니다.
Action: {tool_names}
Action Input: (검색어)
Observation: (결과)

**상황 2. 정보가 충분할 경우 (답변 작성):**
Question: (질문)
Thought: 내부 지침에서 충분한 정보를 확인했습니다. (이 줄에 답변 쓰지 마세요!)
Final Answer: 첫 문장은 상황에 맞게 유연하게 하세요.

---

[답변 스타일 및 언어 가이드] **(매우 중요)**
1. **절대 한자(Chinese characters)나 중국어를 사용하지 마세요.**
   - 예: '물样' -> '물 같은', '剧痛' -> '심한 통증', '很快' -> '빠르게'
   - 모든 전문 용어는 **한글**로 풀어서 쓰세요.
2. **자연스러운 한국어 사용**:
   - 기계적인 번역투를 피하고, 동네 수의사 선생님처럼 자연스럽게 말하세요.
3. **가독성**:
   - 번호(1., 2.)와 볼드체를 적극 활용하세요.

[진단 요약]
{context}

[원내 의학 지침 (우선 참고)]
{local_knowledge}

[이전 대화]
{chat_history}

---
위 규칙을 철저히 지켜 답변하세요. 특히 **Final Answer:** 뒤에 한글 답변을 작성하세요.

Question: {input}
Thought: {agent_scratchpad}
"""
        self.prompt = PromptTemplate.from_template(self.template)
        # Pod 환경 표준: langchain_classic 사용
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True, max_iterations=3)

    def start_case(self, case_id, diagnosis, report_text, symptoms=None, image_path=None):
        return DogEyeCase(case_id, diagnosis, report_text, symptoms or [], image_path)

    def answer(self, case, question, chat_history_str):
        diag_info = f"""
- 진단명: {case.diagnosis}
- 증상: {', '.join(case.symptoms)}
- 내부 리포트 요약: {case.report_text[:500]}...
"""
        local_text = load_local_knowledge(case.diagnosis)
        try:
            response = self.agent_executor.invoke({
                "input": question,
                "context": diag_info,
                "local_knowledge": local_text,
                "chat_history": chat_history_str
            })
            return response["output"]
        except Exception as e:
            print(f"Agent Error: {e}")
            return "죄송해요, 잠시 후 다시 시도해 주세요. 😢"