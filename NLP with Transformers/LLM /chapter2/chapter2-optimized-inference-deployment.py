# 7. 최적화된 추론 배포
"""

● 최적화된 추론 배포 프레임워크 정리

  주요 프레임워크별 특징

  1. TGI (Text Generation Inference)

  - 메모리 관리: 고정된 시퀀스 길이로 일관된 메모리 사용
  - 핵심 기술:
    - Flash Attention 2: 어텐션 계산 최적화
    - 연속 배칭(CBA): GPU 활용도 극대화
  - 장점: 프로덕션 환경에서 안정적이고 예측 가능

  2. vLLM

  - 메모리 관리: PagedAttention 기술 사용
  - 핵심 기술:
    - 메모리를 작은 "페이지" 단위로 분할
    - 비연속 메모리 저장으로 유연한 할당
    - 여러 요청 간 메모리 공유 가능
  - 성능: 기존 대비 최대 24배 높은 처리량
  - 장점: 메모리 단편화 감소, 다양한 크기 요청 효율적 처리

  3. llama.cpp

  - 타겟 환경: 소비자 하드웨어, CPU 중심
  - 핵심 기술:
    - 고도의 양자화 (8비트, 4비트, 3비트, 2비트)
    - GGML/GGUF 형식 사용
    - CPU 아키텍처별 최적화 (AVX2, AVX-512, NEON)
  - 장점: 메모리 제한 환경에서 대규모 모델 실행 가능

  핵심 최적화 기술 설명

  Flash Attention

  - 문제점: 기존 어텐션은 HBM↔SRAM 간 반복적 데이터 전송으로 병목
  - 해결책: 데이터를 SRAM에 한 번만 로드하고 모든 계산을 SRAM에서 수행
  - 효과: 메모리 전송 최소화, GPU 유휴 시간 감소

  PagedAttention (vLLM)

  - 문제점: KV 캐시가 길어질수록 메모리 부담 증가
  - 해결책:
    - KV 캐시를 페이지 단위로 분할
    - 페이지 테이블로 관리
    - 필요시 페이지 공유
  - 효과: 메모리 효율성 극대화

  양자화 (llama.cpp)

  - 목적: 모델 크기와 메모리 사용량 감소
  - 방법: 32/16비트 → 8/4/3/2비트로 정밀도 축소
  - 효과: 품질 손실 최소화하며 추론 속도 향상

  프레임워크 선택 가이드

  - TGI: 안정적인 프로덕션 서버 환경
  - vLLM: 높은 처리량이 필요한 대규모 서비스
  - llama.cpp: 개인 PC나 엣지 디바이스에서 실행
"""

# Hugging Face의 InferenceClient
from huggingface_hub import InferenceClient

# Initialize client pointing to TGI endpoint
client = InferenceClient(
    model="http://localhost:8080",  # URL to the TGI server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[],
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# OpenAI 클라이언트

from openai import OpenAI # TGI 엔드포인트를 가리키는 클라이언트 초기화 
client = OpenAI( 
    base_url= "http://localhost:8080/v1" ,   # /v1을 포함해야 함 
    api_key= "not-needed" ,   # TGI는 기본적으로 API 키가 필요하지 않음 
) # 채팅 완료 
response = client.chat.completions.create( 
    model= "HuggingFaceTB/SmolLM2-360M-Instruct" , 
    messages=[ 
        { "role" : "system" , "content" : "당신은 도움이 되는 조수입니다." }, 
        { "role" : "user" , "content" : "이야기를 들려주세요" }, 
    ], 
    max_tokens= 100 , 
    temperature= 0.7 , 
    top_p= 0.95 , 
) print (response.choices[ 0 ].message.content)

