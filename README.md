# Runpod Serverless - Qwen 2.5-VL (LoRA) Minimal Worker

이 레포는 **Runpod Serverless**에서 **Qwen 2.5-VL**(LoRA 포함)을 돌리기 위한 최소 파일 셋입니다.  
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

## 3) 테스트 (Run → /runsync)
```json
{
  "input": {
    "prompt": "[diagnosis_report] 사진을 보고 진단 리포트를 작성하세요...",
    "images": ["https://example.com/dog_eye.jpg"],
    "gen": {"max_new_tokens": 512, "temperature": 0.2}
  }
}
