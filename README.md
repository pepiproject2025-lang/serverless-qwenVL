# Runpod Serverless - Qwen 3-VL (LoRA) Minimal Worker

이 레포는 **Runpod Serverless**에서 **Qwen 3-VL**(LoRA 포함)을 돌리기 위한 파일 셋입니다.  
**모델/체크포인트는 Runpod Storage(네트워크 볼륨)**에 있고, 이 컨테이너는 런타임에 그 볼륨을 마운트하여 로드합니다.

## 1) 필요한 환경 변수
- `MODEL_BASE` (예: `/runpod-volume/models/Qwen2.5-VL-7B-Instruct`)
- `LORA_PATH`  (예: `/runpod-volume/checkpoints/qwen25vl_lora`) — LoRA 없으면 비워도 OK
- `LOAD_4BIT`  (`true`/`false`, 기본 `true`)
- `MERGE_LORA` (`true`/`false`, 기본 `false`) — 배포 고정 후 속도/메모리 최적화용

## 2) Runpod 콘솔 설정
- **Serverless → New Endpoint → Import from GitHub**
- Repository/Branch/Dockerfile Path 선택
- **Volumes**: Storage 볼륨 → Mount Path: `/runpod-volume`
- **Env**: 위 환경변수 등록
- **Resources**: GPU(L4/A10G/A100), Min Workers=0, Max=1~2, Timeout=600s

## 3) 파일 설명
- **eye_analysis_module.py**: 진단 및 진단 정보를 토데로 보고서 형태의 설명을 만들어주는 파일
- **eye_rag_chatbot2.py**: 진단 결과를 통해 rag와 websearch를 포함한 모델 답변을 생성 해주는 파일
- **handler.py**: 위 2개의 파일과 런팟 서버리스를 연동해 학습한 모델과 통신이 가능하게 끔 해주는 파일
