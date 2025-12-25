#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye_rag_chatbot2.py
-------------------
LangChain ReAct Agent 기반의 챗봇 모듈 (Qwen3-VL + DuckDuckGo + Local Corpus)
"""

import os
import torch
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field

# LangChain & HuggingFace imports
from transformers import AutoModelForVision2Seq, AutoProcessor
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
#from langchain_classic.agents import create_react_agent, AgentExecutor
# langchain_classic이 아니라 langchain.agents를 사용해야 합니다.
from langchain.agents import create_react_agent, AgentExecutor

# ----------------------------------
# 1) 전역 설정 및 모델 캐싱 (Cold Start 방지)
# ----------------------------------

MODEL_DIR = "/workspace/models/Qwen3_VL_8B_Instruct"  # 경로 확인 필요
CORPUS_DIR = "/workspace/corpus/"

# 전역 변수로 모델을 잡아두어 핸들러가 재호출될 때 리로딩 방지
_GLOBAL_MODEL = None
_GLOBAL_PROCESSOR = None

def load_global_model():
    global _GLOBAL_MODEL, _GLOBAL_PROCESSOR
    if _GLOBAL_MODEL is not None:
        return _GLOBAL_MODEL, _GLOBAL_PROCESSOR

    print(f"Loading Model from {MODEL_DIR}...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
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
# 2) Custom LLM Wrapper (노트북 코드 적용)
# ----------------------------------
class QwenVLLLM(LLM):
    model: Any = None
    processor: Any = None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Qwen-VL 채팅 포맷 적용
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(
            text=[text], images=None, videos=None, padding=True, return_tensors="pt"
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": 1024, # 답변 길이 확보
            "do_sample": True,
            "temperature": 0.1,     # 사실 기반 답변을 위해 낮춤
            "repetition_penalty": 1.1,
            "top_p": 0.9,
            **kwargs
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Stop Token 처리
        if stop:
            for s in stop:
                if s in output_text:
                    output_text = output_text.split(s)[0]
        return output_text

    @property
    def _llm_type(self) -> str:
        return "qwen-vl-custom"

# ----------------------------------
# 3) Local Knowledge Loader (노트북 코드 적용)
# ----------------------------------
def load_local_knowledge(diagnosis_name: str) -> str:
    """
    진단명(예: 결막염)을 입력받아 /workspace/corpus/결막염.txt 파일을 읽어서 반환합니다.
    """
    filename = f"{diagnosis_name}.txt"
    filepath = os.path.join(CORPUS_DIR, filename)
    
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"[System] 내부 문서 로드 성공: {filename}")
            return content
        else:
            print(f"[System] 내부 문서 없음: {filename}")
            return "해당 질환에 대한 내부 상세 지침 파일이 존재하지 않습니다."
            
    except Exception as e:
        return f"내부 문서 로딩 중 오류 발생: {e}"

# ----------------------------------
# 4) EyeRAGChatbot2 Class (LangChain Agent Encapsulation)
# ----------------------------------
@dataclass
class DogEyeCase:
    case_id: str
    diagnosis: str
    report_text: str
    symptoms: List[str] = field(default_factory=list)

class EyeRAGChatbot2:
    def __init__(self):
        # 1. 모델 로드 (전역 캐시 활용)
        model, processor = load_global_model()
        self.llm = QwenVLLLM(model=model, processor=processor)

        # 2. 도구 설정 (DuckDuckGo HTML backend - 차단 우회)
        wrapper = DuckDuckGoSearchAPIWrapper(
            backend="html", 
            max_results=5
        )
        self.search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
        self.tools = [self.search_tool]

        # 3. 프롬프트 템플릿 정의 (노트북 최신 버전)
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
   - 기계적인 번역투("당신의 사랑받는 반려견을 위해...")를 피하세요.
   - 실제 한국 동물병원 수의사 선생님처럼 **"~해 주시는 게 좋아요", "~일 가능성이 높아요"** 처럼 자연스럽게 말하세요.
3. **가독성**:
   - 줄글보다는 **번호(1., 2.)**를 사용해 정리해 주세요.
   - 핵심 내용은 **볼드체**로 강조하세요.
   - 이모지를 적절히 사용하여(1~2개 정도) 딱딱하지 않게 해 주세요.

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

        # 4. 에이전트 생성
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
        )

    def start_case(
        self,
        case_id: str,
        diagnosis: str,
        report_text: str,
        symptoms: Optional[List[str]] = None,
        image_path: Optional[str] = None, # 호환성 유지용
    ) -> DogEyeCase:
        """
        새로운 케이스 정보를 생성합니다.
        """
        return DogEyeCase(
            case_id=case_id,
            diagnosis=diagnosis,
            report_text=report_text,
            symptoms=symptoms or []
        )

    def answer(self, case_id: str, question: str, case: DogEyeCase, chat_history_str: str) -> str:
        """
        LangChain Agent를 실행하여 답변을 생성합니다.
        """
        # Context 구성 (진단 정보 + 리포트 내용)
        diag_info = f"""
- 진단명: {case.diagnosis}
- 증상: {', '.join(case.symptoms)}
- 내부 리포트 요약: {case.report_text[:500]}...
"""
        # Local Knowledge 로드
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
            print(f"Agent Execution Error: {e}")
            return "죄송해요, 답변을 생성하는 도중 문제가 발생했어요. 잠시 후 다시 시도해 주세요. 😢"