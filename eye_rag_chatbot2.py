#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye_rag_chatbot2.py
-------------------
기존 eye_rag_chatbot.py 를 함수 중심으로 최대한 모듈화한 단일 파일 버전입니다.

구성 개요
    - 설정/상수: AppConfig, 진단명/별칭/시드 KB
    - 외부 검색: DuckDuckGo, Wikipedia 래퍼 함수
    - RAG 유틸: 도메인 필터, 디듀프, 로컬 코퍼스 로딩, 리포트 슬라이스, 컨텍스트 빌드
    - Qwen3-VL 클라이언트: 로컬 모델 + OpenAI 호환 엔드포인트
    - 챗봇 상태/함수: ChatbotState, start_case(), answer_question(), 등
    - 얇은 클래스 래퍼: EyeRAGChatbot2 (함수들을 감싸는 편의용)
    - 간단 CLI: demo_cli()
"""

from __future__ import annotations

import os
import re
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ddgs import DDGS  # type: ignore[import]
except Exception:  # pragma: no cover - optional
    print("DDGS not found")
    DDGS = None  # type: ignore[assignment]

try:
    import wikipedia  # type: ignore[import]
    wikipedia.set_lang("en")
except Exception:  # pragma: no cover - optional
    print("Wikipedia not found")
    wikipedia = None  # type: ignore[assignment]

#from openai import OpenAI


# ----------------------------------
# 1) 설정/상수 (AppConfig + 진단명/별칭/시드 KB)
# ----------------------------------

PREFERRED_DOMAINS: List[str] = [
    "merckvetmanual.com",
    "vcahospitals.com",
    "cornell.edu",
    "avma.org",
    "aaha.org",
    "acvo.org",
    "vin.com",
]


DIAGNOSES: List[str] = ["결막염", "궤양성각막질환", "백내장", "안검염", "안검내반증", "유루증"]


ALIASES: Dict[str, List[str]] = {
    "결막염": ["conjunctivitis dog", "canine conjunctivitis", "conjunctiva dog"],
    "궤양성각막질환": [
        "corneal ulcer dog",
        "ulcerative keratitis dog",
        "canine corneal ulcer",
    ],
    "백내장": ["cataract dog", "canine cataract", "lens opacity dog"],
    "안검염": [
        "blepharitis dog",
        "canine blepharitis",
        "eyelid inflammation dog",
    ],
    "안검내반증": ["entropion dog", "eyelid entropion dog"],
    "유루증": ["epiphora dog", "tear staining dog", "excessive tearing dog"],
}


SEED_KB: Dict[str, List[Dict[str, str]]] = {
    "결막염": [
        {
            "id": "seed_merck_conj",
            "title": "Merck Vet Manual — Conjunctiva (dogs)",
            "text": "반려견 결막염은 감염, 알레르기, 자극 요인 등으로 결막 충혈·분비물 증가가 흔합니다. "
            "임상에서는 충혈, 부종(chemosis), 분비물 성상(수양/점액/고름)을 평가합니다.",
            "source": "seed",
            "url": "https://www.merckvetmanual.com/eye-diseases-and-disorders/ophthalmology/conjunctiva",
        },
        {
            "id": "seed_vca_conj",
            "title": "VCA — Conjunctivitis in Dogs",
            "text": "결막 염증은 통증/깜빡임 증가와 함께 분비물·충혈을 동반할 수 있습니다. "
            "원인 규명과 위생 관리가 중요합니다.",
            "source": "seed",
            "url": "https://vcahospitals.com/know-your-pet/conjunctivitis-in-dogs",
        },
    ],
    "궤양성각막질환": [
        {
            "id": "seed_merck_ulcer",
            "title": "Merck Vet Manual — Corneal Ulcers (dogs)",
            "text": "각막 궤양은 표면 결손과 통증, 눈물 과다, 혼탁이 특징입니다. "
            "지연 시 감염 악화·용해성 궤양·천공 위험이 큽니다.",
            "source": "seed",
            "url": "https://www.merckvetmanual.com/eye-diseases-and-disorders/corneal-disease/corneal-ulcers-in-dogs",
        },
        {
            "id": "seed_vca_ulcer",
            "title": "VCA — Corneal Ulcers in Dogs",
            "text": "범위는 표층부터 심부 궤양까지 다양하며 형광염색 검사가 유용합니다. "
            "신속한 처치가 예후에 중요합니다.",
            "source": "seed",
            "url": "https://vcahospitals.com/know-your-pet/corneal-ulcers-in-dogs",
        },
    ],
    "백내장": [
        {
            "id": "seed_cornell_cat",
            "title": "Cornell — Canine Cataracts",
            "text": "백내장은 수정체 혼탁으로 시력 저하를 유발합니다. 노령·유전·대사성 요인 등 "
            "원인이 다양하며 핵경화와 감별이 필요합니다.",
            "source": "seed",
            "url": "https://www.vet.cornell.edu/departments-centers-and-institutes/riney-canine-health-center/canine-health-information/canine-cataracts",
        },
        {
            "id": "seed_vca_cat",
            "title": "VCA — Cataracts in Dogs",
            "text": "동공 뒤 혼탁과 반사 저하가 단서이며, 진행도에 따라 관리·수술 여부를 결정합니다.",
            "source": "seed",
            "url": "https://vcahospitals.com/know-your-pet/cataracts-in-dogs",
        },
    ],
    "안검염": [
        {
            "id": "seed_vca_bleph",
            "title": "VCA — Blepharitis in Dogs",
            "text": "안검 가장자리 염증으로 발적·비후·딱지·가려움이 흔합니다. "
            "원인 다양하며 위생 및 원인 치료가 핵심입니다.",
            "source": "seed",
            "url": "https://vcahospitals.com/know-your-pet/blepharitis-in-dogs",
        }
    ],
    "안검내반증": [
        {
            "id": "seed_vca_entro",
            "title": "VCA — Eyelid Entropion in Dogs",
            "text": "안검 내반은 속눈썹/피부가 각막을 문질러 궤양 위험을 높입니다. "
            "중증은 수술 교정이 권장됩니다.",
            "source": "seed",
            "url": "https://vcahospitals.com/know-your-pet/eyelid-entropion-in-dogs",
        }
    ],
    "유루증": [
        {
            "id": "seed_vca_epi",
            "title": "VCA — Eye Discharge (Epiphora) in Dogs",
            "text": "눈물 배출 장애 또는 과다 분비로 눈가 젖음과 착색이 생깁니다. "
            "원인에 따라 관리와 치료가 달라집니다.",
            "source": "seed",
            "url": "https://vcahospitals.com/know-your-pet/eye-discharge-or-epiphora-in-dogs",
        }
    ],
}


@dataclass
class AppConfig:
    """환경설정/파라미터를 한 곳에서 관리하기 위한 설정 객체."""

    # 검색/코퍼스 관련
    use_ddg: bool = True
    use_wiki: bool = True
    k: int = 12
    wiki_pages: int = 1
    corpus_dir: str = "./corpus"

    # Qwen 관련
    qwen_local_model_dir: str = "/workspace/models/Qwen3_VL_8B_Instruct"
    # qwen_api_key: Optional[str] = None
    # qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # qwen_model: str = "qwen-vl-max"

    # 프롬프트/대화 히스토리 관련
    answer_max_lines: int = 5
    max_history_turns: int = 4

    @classmethod
    def from_env(cls) -> "AppConfig":
        """환경변수에서 설정을 읽어 AppConfig 생성."""
        return cls(
            use_ddg=os.getenv("EYE_RAG_USE_DDG", "1") == "1",
            use_wiki=os.getenv("EYE_RAG_USE_WIKI", "1") == "1",
            k=int(os.getenv("EYE_RAG_K", "12")),
            wiki_pages=int(os.getenv("EYE_RAG_WIKI_PAGES", "1")),
            corpus_dir=os.getenv("EYE_RAG_CORPUS_DIR", "./corpus"),
            qwen_local_model_dir=os.getenv(
                "QWEN_LOCAL_MODEL_DIR",
                "/workspace/models/Qwen3_VL_8B_Instruct",
            ),
            # qwen_api_key=os.getenv("QWEN_API_KEY") or None,
            # qwen_api_base=os.getenv(
            #     "QWEN_API_BASE",
            #     "https://dashscope.aliyuncs.com/compatible-mode/v1",
            # ),
            # qwen_model=os.getenv("QWEN_MODEL", "qwen-vl-max"),
            answer_max_lines=int(os.getenv("EYE_RAG_ANSWER_MAX_LINES", "5")),
            max_history_turns=int(os.getenv("EYE_RAG_MAX_HISTORY_TURNS", "4")),
        )


# ----------------------------------
# 2) 데이터 모델
# ----------------------------------


@dataclass
class DogEyeCase:
    case_id: str
    diagnosis: str
    report_text: str
    image_path: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ChatbotState:
    """챗봇 인스턴스의 상태를 보관하는 단순 컨테이너."""
    config: AppConfig
    local_by_diag: Dict[str, List[Dict[str, Any]]]
    cases: Dict[str, DogEyeCase] = field(default_factory=dict)


# ----------------------------------
# 3) 외부 검색 (DuckDuckGo, Wikipedia)
# ----------------------------------


def ddg_search(queries: List[str], max_results: int = 6) -> Tuple[List[Dict[str, Any]], List[str]]:
    """DuckDuckGo로 텍스트 검색."""
    results: List[Dict[str, Any]] = []
    used_queries: List[str] = []
    if DDGS is None:
        return results, used_queries
    try:
        with DDGS() as ddgs:  # type: ignore[call-arg]
            for q in queries:
                got_any = False
                for r in ddgs.text(q, max_results=max_results):
                    url = r.get("href") or r.get("url") or ""
                    title = r.get("title") or ""
                    body = r.get("body") or ""
                    if not url or "bing.com/aclick" in url:
                        continue
                    results.append(
                        {
                            "id": f"ddg::{hash((q, url)) & 0xFFFF_FFFF:x}",
                            "title": title[:200],
                            "text": (body or "")[:2000],
                            "url": url,
                            "source": "ddg",
                            "q": q,
                        }
                    )
                    got_any = True
                if got_any:
                    used_queries.append(q)
    except Exception as e:  # pragma: no cover - 네트워크 에러
        print("[ddg] error:", repr(e))
    return results, used_queries


def wiki_chunks(query: str, max_pages: int = 1, max_chars: int = 2500) -> List[Dict[str, Any]]:
    """Wikipedia에서 검색한 페이지 내용을 잘라서 반환."""
    out: List[Dict[str, Any]] = []
    if wikipedia is None:
        return out
    try:
        titles = wikipedia.search(query, results=max_pages)  # type: ignore[call-arg]
        for t in titles or []:
            try:
                page = wikipedia.page(title=t, auto_suggest=False, redirect=True)  # type: ignore[call-arg]
                text = page.content[:max_chars]
                out.append(
                    {
                        "id": f"wiki::{hash((t, query)) & 0xFFFF_FFFF:x}",
                        "title": page.title,
                        "text": text,
                        "url": page.url,
                        "source": "wiki",
                        "q": query,
                    }
                )
            except Exception:
                continue
    except Exception as e:  # pragma: no cover - 네트워크 에러
        print("[wiki] error:", repr(e))
    return out


# ----------------------------------
# 4) RAG 유틸 함수 (도메인 필터/코퍼스/컨텍스트 빌드)
# ----------------------------------


def domain_ok(url: str) -> bool:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        return any(host.endswith(d) for d in PREFERRED_DOMAINS)
    except Exception:
        return False


def filter_docs(docs: List[Dict[str, Any]], min_len: int = 120) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs:
        txt = (d.get("text") or "").strip()
        url = (d.get("url") or "").strip()
        if len(txt) < min_len:
            continue
        if url and "aclick" in url:
            continue
        out.append(d)
    return out


def dedup_near_duplicates(docs: List[Dict[str, Any]], by_url: bool = True) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for d in docs:
        key = d.get("url") if (by_url and d.get("url")) else (d.get("title") or "")[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def prefer_authority(docs: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for d in docs:
        url = d.get("url") or ""
        score = 0
        if domain_ok(url):
            score += 5
        if d.get("source") == "ddg":
            score += 1
        if d.get("source") == "wiki":
            score += 1
        if "dog" in (d.get("title") or "").lower():
            score += 1
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_n]]


def inject_seed_if_needed(docs: List[Dict[str, Any]], diagnosis: str) -> List[Dict[str, Any]]:
    """웹 검색이 잘 안 나왔을 경우 시드 KB를 주입."""
    if any(d.get("url") for d in docs):
        return docs
    return (SEED_KB.get(diagnosis, [])[:2] + docs)


def load_local_corpus_by_diag(corpus_dir: str = "./corpus") -> Dict[str, List[Dict[str, Any]]]:
    """
    corpus/*.txt 중 파일명에 진단명이 포함된 것만 해당 진단의 로컬 코퍼스로 사용.
    첫 줄: 제목, 이후: 본문.
    """
    by_diag: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DIAGNOSES}
    p = Path(corpus_dir)
    if not p.exists():
        return by_diag

    # NFC 정규화 (윈도우/맥 파일명 깨짐 방지)
    for q in p.glob("*"):
        nfc = unicodedata.normalize("NFC", q.name)
        if nfc != q.name:
            q.rename(q.with_name(nfc))

    for fp in p.glob("*.txt"):
        name = fp.stem
        target_diag: Optional[str] = None
        for d in DIAGNOSES:
            if d in name:
                target_diag = d
                break
        if not target_diag:
            continue
        try:
            text = fp.read_text(encoding="utf-8").strip()
            if not text:
                continue
            lines = text.splitlines()
            title = lines[0][:200] if lines else fp.stem
            body = "\n".join(lines[1:]) if len(lines) > 1 else ""
            by_diag[target_diag].append(
                {
                    "id": f"local::{fp.stem}",
                    "title": title,
                    "text": body[:6000],
                    "url": None,
                    "source": "local",
                }
            )
        except Exception:
            continue
    return by_diag


def split_report_into_docs(case: DogEyeCase, max_chars: int = 900) -> List[Dict[str, Any]]:
    """진단 리포트 텍스트를 여러 doc으로 쪼개서 RAG에 사용."""
    text = (case.report_text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    docs: List[Dict[str, Any]] = []
    idx = 1
    for p in paragraphs:
        chunk = p
        while chunk:
            docs.append(
                {
                    "id": f"case::{case.case_id}::{idx}",
                    "title": f"진단 리포트 요약 {idx}",
                    "text": chunk[:max_chars],
                    "url": None,
                    "source": "case",
                }
            )
            chunk = chunk[max_chars:]
            idx += 1
    return docs


def build_queries(case: DogEyeCase, question: str) -> List[str]:
    """진단명 + 보호자 질문으로 검색 쿼리 구성."""
    base_aliases = ALIASES.get(case.diagnosis, [])
    q_kr = f"{case.diagnosis} 반려견 {question}"
    q_en = ""
    if base_aliases:
        q_en = base_aliases[0] + " " + question
    queries = [q for q in [q_kr, q_en] + base_aliases[:2] if q]
    # 중복 제거
    uniq: List[str] = []
    for q in queries:
        if q not in uniq:
            uniq.append(q)
    return uniq


def simple_keyword_score(text: str, question: str) -> int:
    """아주 간단한 키워드 기반 스코어."""
    score = 0
    for token in question.split():
        if len(token) < 2:
            continue
        if token.lower() in text.lower():
            score += 1
    return score


def build_rag_context(
    case: DogEyeCase,
    question: str,
    local_docs: List[Dict[str, Any]],
    config: AppConfig,
) -> List[Dict[str, Any]]:
    """
    케이스 리포트 + 로컬 코퍼스 + 웹검색 결과를 합쳐 상위 k개 RAG 컨텍스트로 반환.
    케이스/로컬 정보를 우선적으로 반영하도록 간단한 스코어링을 추가.
    """
    merged: List[Dict[str, Any]] = []

    # 1) 케이스 리포트
    case_docs = split_report_into_docs(case)
    merged += case_docs

    # 2) 로컬 코퍼스 (진단별 txt)
    merged += local_docs

    # 3) 웹검색 (DDG + Wiki)
    queries = build_queries(case, question)
    if config.use_ddg:
        ddg_docs, _ = ddg_search(queries, max_results=6)
        merged += ddg_docs

    if config.use_wiki:
        wiki_docs: List[Dict[str, Any]] = []
        for q in queries[:2]:
            wiki_docs += wiki_chunks(q, max_pages=config.wiki_pages)
        merged += wiki_docs

    # 4) 시드 KB 보강
    merged = inject_seed_if_needed(merged, case.diagnosis)

    # 5) 필터/스코어/디듀프
    merged = filter_docs(merged, min_len=120)
    merged = prefer_authority(merged, top_n=max(config.k * 2, 12))
    merged = dedup_near_duplicates(merged, by_url=True)

    # 6) 질문 기반 스코어 + 출처 기반 가중치
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for d in merged:
        base_score = 0
        if d.get("source") == "case":
            base_score += 5
        if d.get("source") == "local":
            base_score += 3
        kw_score = simple_keyword_score(d.get("text") or "", question)
        scored.append((base_score + kw_score, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    final_docs = [d for _, d in scored[: config.k]]
    return final_docs


def build_ctx_block(ctx_docs: List[Dict[str, Any]]) -> str:
    """모델에게 줄 RAG_CONTEXT 텍스트 블록 생성."""
    lines: List[str] = []
    for i, d in enumerate(ctx_docs, 1):
        lines.append(
            f"[{i}] {d.get('title') or 'Untitled'} :: {d.get('url') or 'no-url'}\n"
            f"{(d.get('text') or '')[:1200]}"
        )
    return "\n\n".join(lines)


# ----------------------------------
# 5) Qwen3-VL 클라이언트 (로컬 + 원격)
# ----------------------------------

_QWEN_LOCAL_TOKENIZER = None
_QWEN_LOCAL_PROCESSOR = None
_QWEN_LOCAL_MODEL = None


def _ensure_local_qwen_loaded(config: AppConfig) -> None:
    """로컬 Qwen-VL 모델/토크나이저를 lazy-load."""
    global _QWEN_LOCAL_TOKENIZER, _QWEN_LOCAL_PROCESSOR, _QWEN_LOCAL_MODEL
    if _QWEN_LOCAL_MODEL is not None and _QWEN_LOCAL_TOKENIZER is not None:
        return
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
    import torch  # noqa: F401

    model_dir = config.qwen_local_model_dir
    _QWEN_LOCAL_PROCESSOR = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    try:
        _QWEN_LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
    except Exception:
        # processor에 tokenizer가 포함되어 있을 수 있음
        _QWEN_LOCAL_TOKENIZER = getattr(_QWEN_LOCAL_PROCESSOR, "tokenizer", None)
    _QWEN_LOCAL_MODEL = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()


def _call_qwen_local(messages: List[Dict[str, Any]], config: AppConfig) -> str:
    """로컬 Qwen-VL 모델로 추론 시도."""
    if not os.path.isdir(config.qwen_local_model_dir):
        raise RuntimeError("local model dir not found")

    _ensure_local_qwen_loaded(config)
    tokenizer = _QWEN_LOCAL_TOKENIZER
    processor = _QWEN_LOCAL_PROCESSOR
    model_obj = _QWEN_LOCAL_MODEL

    # VL 포맷으로 메시지 변환(텍스트만 사용)
    vl_msgs: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            vl_msgs.append(
                {
                    "role": m.get("role", "user"),
                    "content": [{"type": "text", "text": content}],
                }
            )
        else:
            vl_msgs.append(m)

    # chat template 적용
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(  # type: ignore[call-arg]
            vl_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif hasattr(processor, "apply_chat_template"):
        prompt_text = processor.apply_chat_template(  # type: ignore[call-arg]
            vl_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # fallback: 단순 연결
        prompt_text = "\n".join(
            [f"{m.get('role', 'user')}: {m.get('content')}" for m in messages]
        )

    # 토크나이즈
    if tokenizer is not None:
        inputs = tokenizer(prompt_text, return_tensors="pt")
    else:
        # processor에 tokenizer가 반드시 있다고 가정
        inputs = processor.tokenizer(prompt_text, return_tensors="pt")  # type: ignore[union-attr]

    # device_map="auto"이므로 첫 파라미터 디바이스로 이동
    device = getattr(model_obj, "device", None)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        eos_token_id=(getattr(tokenizer, "eos_token_id", None)),
    )

    import torch  # type: ignore[import]  # noqa: F401

    with torch.no_grad():  # type: ignore[name-defined]
        output_ids = model_obj.generate(**inputs, **gen_kwargs)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]

    # 생성된 전체에서 프롬프트 부분 제거
    if output_text.startswith(prompt_text):
        output_text = output_text[len(prompt_text) :].strip()
    return (output_text or "").strip()


# def _call_qwen_remote(messages: List[Dict[str, Any]], config: AppConfig) -> str:
#     """OpenAI 호환 엔드포인트 호출."""
#     api_key = config.qwen_api_key
#     if not api_key:
#         raise RuntimeError("QWEN_API_KEY is not set and local model unavailable")

#     client = OpenAI(api_key=api_key, base_url=config.qwen_api_base)
#     resp = client.chat.completions.create(
#         model=config.qwen_model,
#         messages=messages,
#         temperature=0.4,
#         timeout=60,
#     )
#     return (resp.choices[0].message.content or "").strip()


def call_qwen3_vl(messages: List[Dict[str, Any]], config: AppConfig) -> str:
    """
    Qwen3-VL 호출 우선순위:
      1) 로컬 모델 디렉토리에서 Transformers로 추론
      2) 실패 시 OpenAI 호환 엔드포인트로 폴백
    """
    # 1) 로컬 모델 우선 시도
    try:
        return _call_qwen_local(messages, config)
    except Exception:
        traceback.print_exc()

    # 2) 원격 폴백
    #return _call_qwen_remote(messages, config)


# ----------------------------------
# 6) 챗봇 상태/함수 (start_case, compose_messages, answer_question)
# ----------------------------------

ANSWER_STYLE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "brief": {
        "instruction": (
            "질문에 대한 핵심 답을 3~5문장 정도의 짧은 단락으로 정리해 주세요. "
            "목록을 쓰더라도 항목 수는 3개 이내로 제한해 주세요."
        ),
        "max_lines": 5,
    },
    "detail": {
        "instruction": (
            "질문에 대한 직접적인 답변 → 왜 그런지 이유 설명 → 집에서 할 수 있는 "
            "구체적인 관리 방법과 주의사항을 순서대로 자세히 설명해 주세요. "
            "필요하다면 번호 목록을 활용해도 좋아요."
        ),
        "max_lines": 20,
    },
}


def create_chatbot_state(config: Optional[AppConfig] = None) -> ChatbotState:
    """기본 설정/로컬 코퍼스를 로딩한 ChatbotState 생성."""
    cfg = config or AppConfig.from_env()
    local_by_diag = load_local_corpus_by_diag(cfg.corpus_dir)
    return ChatbotState(config=cfg, local_by_diag=local_by_diag)


def start_case(
    state: ChatbotState,
    case_id: str,
    diagnosis: str,
    report_text: str,
    image_path: Optional[str] = None,
    symptoms: Optional[List[str]] = None,
) -> DogEyeCase:
    """새 케이스를 생성/등록."""
    diagnosis = (diagnosis or "").strip()
    if not diagnosis:
        raise ValueError("diagnosis is required")

    case = DogEyeCase(
        case_id=case_id,
        diagnosis=diagnosis,
        report_text=report_text.strip(),
        image_path=image_path,
        symptoms=symptoms or [],
    )
    state.cases[case_id] = case
    return case


def compose_messages(
    state: ChatbotState,
    case: DogEyeCase,
    question: str,
    ctx_docs: List[Dict[str, Any]],
    mode: str = "brief",
) -> List[Dict[str, Any]]:
    """
    Qwen3-VL에 전달할 messages 생성.
    - system: 역할/안전 규칙/컨텍스트 사용 방법
    - user: 보호자 질문 + 케이스 정보
    - + 기존 대화 히스토리 일부 포함
    모드(brief/detail)에 따라 답변 길이/스타일을 조정.
    """
    style = ANSWER_STYLE_TEMPLATES.get(mode, ANSWER_STYLE_TEMPLATES["brief"])
    max_lines = style["max_lines"]

    # CTX 블록 생성
    ctx_block = build_ctx_block(ctx_docs)

    # 증상 텍스트
    sym_str = ", ".join(case.symptoms) if case.symptoms else "별도로 정리된 증상 목록 없음"

    # safety + 역할
    system_msg = {
        "role": "system",
        "content": (
            "당신은 반려견 안과 전문 수의사이면서, 보호자에게 친절하게 설명해 주는 상담 챗봇이에요. "
            "아래 [RAG_CONTEXT]에 있는 정보와 케이스 리포트를 근거로만 답변해야 해요.\n\n"
            "안전 규칙:\n"
            "- 실제 진단/치료를 대신하지 않는다는 점을 항상 상기시키세요.\n"
            "- 심한 통증, 시력 상실, 눈이 갑자기 하얗게 변함, 안구가 튀어나온 느낌, 눈을 지속적으로 감고 있는 경우 등은 "
            "반드시 즉시 오프라인 수의사/응급실 방문을 권고하세요.\n"
            "- 약 이름, 용량, 구체적인 처방 변경, 기존 약을 임의로 중단/추가하도록 지시하지 마세요.\n"
            "- 집에서 할 수 있는 관리(눈 주변 청결, 보호 콘 사용, 활동 제한 등)를 위주로 설명하세요.\n\n"
            "답변 스타일:\n"
            "- 한국어로 말하고, 문장을 '...에요', '...해요'처럼 부드럽고 친근한 말투로 작성하세요.\n"
            "- 보호자가 초보라고 가정하고, 어려운 용어는 쉽게 풀어서 설명하세요.\n"
            f"- 아래 지침에 맞게 답변 길이와 구조를 맞춰 주세요.\n"
            f"- 지침: {style['instruction']}\n\n"
            f"[RAG_CONTEXT]\n{ctx_block}\n"
        ),
    }

    # 기존 히스토리에서 최근 N턴만 가져오기
    history_msgs: List[Dict[str, Any]] = []
    max_turns = state.config.max_history_turns
    for turn in case.history[-max_turns:]:
        history_msgs.append({"role": turn["role"], "content": turn["content"]})

    # 유저 질문 메시지
    user_msg = {
        "role": "user",
        "content": (
            f"케이스 정보:\n"
            f"- 진단명: {case.diagnosis}\n"
            f"- 증상 요약: {sym_str}\n"
            f"- 참고: 이미지 경로는 {case.image_path or '미지정'} 이고, "
            f"이미지는 이미 수의사가 검토해서 위 진단이 내려진 상태라고 가정해요.\n\n"
            f"보호자 질문: {question}\n\n"
            "위 케이스를 기준으로 보호자가 집에서 무엇을 할 수 있고, "
            "어떻게 관리해야 할지 위 지침에 맞춰 설명해 주세요.\n"
            f"가능하면 최대 {max_lines}줄 이내로 정리해 주세요."
        ),
    }

    messages: List[Dict[str, Any]] = [system_msg] + history_msgs + [user_msg]
    return messages


def answer_question(
    state: ChatbotState,
    case_id: str,
    question: str,
    mode: str = "brief",
) -> str:
    """
    보호자 질문에 대해 Qwen3-VL을 이용해 답변을 생성.

    mode:
        - "brief": 짧고 핵심 위주
        - "detail": 이유/관리 방법까지 자세히
    """
    if case_id not in state.cases:
        raise KeyError(f"case_id '{case_id}' not found")

    case = state.cases[case_id]

    try:
        local_docs = state.local_by_diag.get(case.diagnosis, [])
        ctx_docs = build_rag_context(case, question, local_docs, config=state.config)
        messages = compose_messages(state, case, question, ctx_docs, mode=mode)
        answer_text = call_qwen3_vl(messages, config=state.config)
    except Exception:
        traceback.print_exc()
        # 폴백: 간단한 안내 메시지
        answer_text = (
            "답변을 생성하는 중에 오류가 발생했어요. "
            "그래도 한 가지 확실한 건, 눈 질환은 악화 속도가 빠를 수 있어서 "
            "통증이 심해 보이거나 눈 모양/색이 급격히 변하면 바로 오프라인 수의사에게 가야 한다는 점이에요."
        )

    # 히스토리에 이번 턴 추가
    case.history.append({"role": "user", "content": question})
    case.history.append({"role": "assistant", "content": answer_text})

    return answer_text


# ----------------------------------
# 7) 얇은 클래스 래퍼 (기존 스타일 호환용)
# ----------------------------------


class EyeRAGChatbot2:
    """
    함수 기반 모듈화를 감싼 얇은 클래스 래퍼.

    사용 예시:
        bot = EyeRAGChatbot2()
        case = bot.start_case(
            case_id="dog_001",
            diagnosis="결막염",
            report_text="...증상/진단 이유/원인/조치...",
            image_path="demo.jpg",
        )
        answer = bot.answer(case.case_id, "집에서 어떻게 관리해야 하나요?")
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.state = create_chatbot_state(config)

    def start_case(
        self,
        case_id: str,
        diagnosis: str,
        report_text: str,
        image_path: Optional[str] = None,
        symptoms: Optional[List[str]] = None,
    ) -> DogEyeCase:
        return start_case(
            self.state,
            case_id=case_id,
            diagnosis=diagnosis,
            report_text=report_text,
            image_path=image_path,
            symptoms=symptoms,
        )

    def answer(self, case_id: str, question: str, mode: str = "brief") -> str:
        return answer_question(self.state, case_id=case_id, question=question, mode=mode)


# ----------------------------------
# 8) 간단 CLI 예제
# ----------------------------------


def demo_cli() -> None:
    """
    python eye_rag_chatbot2.py 를 직접 실행했을 때,
    함수 모듈화 버전 챗봇의 간단 CLI 데모.
    """
    state = create_chatbot_state()

    diagnosis = "결막염"
    report_text = (
        "이 리포트는 예시에요.\n\n"
        "- 증상: 눈곱이 많이 끼고, 눈이 빨갛게 충혈되어 있어요.\n"
        "- 진단 이유: 결막 부위가 붉고 부어 있으며, 분비물 양이 증가했어요.\n"
        "- 원인: 세균/바이러스 감염, 알레르기, 이물 자극 등 여러 가지가 가능해요.\n"
        "- 기본 조치: 수의사 처방 안약을 규칙적으로 넣고, 눈 주변을 깨끗하게 관리하는 것이 중요해요.\n"
    )

    case = start_case(
        state,
        case_id="demo_case",
        diagnosis=diagnosis,
        report_text=report_text,
        image_path="demo_image.jpg",
        symptoms=["충혈", "분비물 증가", "눈 비빔"],
    )

    print("반려견 안구질환 케어 챗봇 (함수 모듈화 버전) 데모를 시작할게요.")
    print("종료하려면 빈 줄을 입력하거나 Ctrl+C 를 누르세요.\n")

    while True:
        try:
            q = input("보호자 질문 > ").strip()
            if not q:
                print("종료합니다.")
                break
            ans = answer_question(state, case.case_id, q, mode="brief")
            print("\n[챗봇 답변]\n")
            print(ans)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break
        except Exception:
            traceback.print_exc()
            print("에러가 발생했어요. 다시 시도해 주세요.\n")


if __name__ == "__main__":
    demo_cli()


