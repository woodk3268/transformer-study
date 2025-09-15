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