# using transformers
# 1. behind the pipeline

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(result)

# 1.1 토크나이저를 사용한 전처리
"""
- 입력을 토큰이라고 하는 단어, 하위 단어 또는 기호(예: 구두점)로 분할
- 각 토큰을 정수로 매핑
- 모델에 유용할 수 있는 추가 입력 추가
"""

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)


# 1.2 모델
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

# 1.2.1 고차원 벡터

"""
Transformer 모듈의 벡터 출력은 일반적으로 크며, 일반적으로 세 가지 차원을 갖습니다.
- 배치 크기 : 한 번에 처리하는 시퀀스 수 (예시에서는 2)
- 시퀀스 길이 : 시퀀스의 숫자 표현 길이 (예시에서는 16)
- 숨겨진 크기 : 각 모델 입력의 벡터 차원 (예시에서는 768)
"""

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# torch.Size([2, 16, 768])

# 1.2.2 Model Head : 숫자로 의미 파악하기
"""
- 모델 헤드는 은닉 상태의 고차원 벡터를 입력으로 받아 다른 차원에 투영
- 일반적으로 하나 또는 여러 개의 선형 레이어로 구성
- Transformer 모델의 출력은 처리를 위해 모델 헤드로 직접 전송됨.

- 모델은 임베딩 계층과 그 이후 계층으로 표현됨. 임베딩 계층은 토큰화된 입력의 각 입력 ID를 연관된 토큰을 나타내는 벡터로 변환
- 이후 계층은 어텐션 메커니즘을 사용하여 이러한 벡터를 조작하여 문장의 최종 표현을 생성
"""

# 모델 헤드는 은닉 상태의 고차원 벡터를 입력으로 받아 다른 차원에 투영.
# 일반적으로 하나 또는 여러 개의 선형 레이어로 구성

from transformers import AutoModelForSequenceClasification

checkpoint = "distilbert-base-uncased-finetuned-set-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# 출력 형태를 살펴보면 차원이 훨씬 낮아짐
# 모델 헤드는 앞서 본 고차원 벡터를 입력으로 받아, 두 개의 값(라벨당 하나씩)을 포함하는 벡터 출력

print(outputs.logits.shape)
# torch.Size([2,2])
# 문장이 2개이고, 레이블도 2개이므로 모델로부터 얻는 결과는 2X2 형태


# 1.3 출력 후처리

print(outputs.logits)

#tensor([[-1.5607,  1.6123],
#        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)

"""모델은 첫 번째 문장에 대해 [-1.5607, 1.6123], 두 번째 문장에 대해 [4.1692, -3.3464] 를 예측.
    이는 확률이 아닌 로짓 값으로, 모델의 마지막 레이어에서 출력되는 정규화되지 않은 원시 점수.
    확률로 변환하려면 소프트맥스 레이어를 거쳐야 함.
    모든 트랜스포머 모델은 로짓 값을 출력. 
    훈련용 손실함수는 일반적으로 소프트맥스와 같은 마지막 활성화 함수를 크로스 엔트로피와 같은 실제 손실 함수와 결합하기 때문.
"""

import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
print(predictions)

#tensor([[4.0195e-02, 9.5980e-01],
#        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)

"""
모델이 첫 번째 문장에 대해 [0.0402, 0.9598], 두 번째 문장에 대해 [0.9995, 0.0005]를 예측
이는 식별 가능한 확률 점수.
각 위치에 해당하는 레이블을 얻으려면 모델 구성의 id2label 확인
"""

# {0: 'NEGATIVE', 1: 'POSITIVE'}
