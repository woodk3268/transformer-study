# 4. 여러 시퀀스 처리

"""
- 여러 개의 시퀀스를 어떻게 처리하나요?
- 길이가 다른 여러 시퀀스를 어떻게 처리하나요?
- vocabulary index가 모델이 잘 작동하도록 하는 유일한 입력인가요?
- 너무 긴 시퀀스가 있을까?
"""

# 4.1 모델은 일괄 입력을 예상

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification 

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# sequence = "I've been waiting for a HuggingFace course my whole life."

# tokens = tokenizer.tokenize(sequence)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# input_ids = torch.tensor(ids)
# model(input_ids)

# IndexError: too many indices for tensor of dimension 1
"""
- 1차원 텐서에 2차원 인덱싱을 시도했기 때문에 발생
- input_ids는 1차원 텐서인데, 모델 내부에서 input_ids[:, [-1,0]] 같은 2차원 인덱싱을 시도하고 있음

- 왜 배치 차원이 필요한가?
: Transformer 모델들은 배치 처리를 기본으로 설계되었음.
    
    - 모델은 입력이 항상 (배치 크기, 시퀀스 길이) 형태일 것으로 기대함
    - 단일 문장도 배치 크기가 1인 배치로 처리해야 함.
    - 이는 효율적인 병렬 처리를 위한 설계

- 해결 방법
    input_ids = torch.tensor([ids]) # 대괄호로 감싸서 2차원으로 만듦
    # 결과 [[101, 1045, 1055, ...]]
    # 형태 (1, sequence_length)

    - 1차원 리스트 ids를 [ids]로 감싸서 2차원 리스트로 만듦
    - torch.tensor() 가 이를 2차원 텐서로 변환함
    - 배치 크기 1, 시퀀스 길이 n인 올바른 입력 형태가 됨
"""


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids])
print("입력 ID : " , input_ids)
output = model(input_ids)
print("logits : ", output.logits)

"""
입력 ID :  tensor([[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
          2026,  2878,  2166,  1012]])
logits :  tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward0>)

"""

# 두 개 이상의 문장을 배치로 묶을 때 문장의 길이가 서로 다르다면?

# 4.2 입력 패딩
# 서로 다른 문장 길이
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
 
# 패딩 사용
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]

# 모델에 전송
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

"""
텐서([[ 1.5694 , - 1.3895 ]], grad_fn=<뒤로 추가>) 
텐서([[ 0.5803 , - 0.4125 ]], grad_fn=<뒤로 추가>) 
텐서([[ 1.5694 , - 1.3895 ], 
        [ 1.3373 , - 1.2163 ]], grad_fn=<뒤로 추가>)
"""

"""
- 두 번째 행은 두 번째 문장의 로짓과 같아야 하지만 값이 완전히 다름
- Transformer 모델의 핵심 특징이 각 토큰의 맥락을 파악하는 어텐션 레이어이기 때문
- 이 레이어는 시퀀스의 모든 토큰을 처리하기 때문에 패딩 토큰을 고려 
- 서로 다른 길이의 개별 문장을 모델에 전달하거나, 동일한 문장에 패딩이 적용된 배치를 전달할 때 동일한 결과를 얻으려면 어텐션 레이어에 패딩 토큰을 무시하도록 설정해야함.
- 이는 어텐션 마스크를 사용하여 수행됨
"""

# 4.2 attention mask
# attention mask는 입력 ID 텐서와 정확히 동일한 모양을 가지며 0과 1로 채워진 텐서.
# 1은 해당 토큰에 주의를 기울여야 하고, 0은 해당 토큰에 주의를 기울이지 않아야 함을 나타냄(즉, 모델의 attention layer에서 무시해야 함)

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1,1,1],
    [1,1,0],
]

outputs = model(torch.tensor(batched_ids), attention_mask = torch.tensor(attention_mask))
print(outputs.logits)

"""
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
"""

# 4.3 더 긴 시퀀스

"""
- Transformer 모델을 사용하면 모델에 전달할 수 있는 시퀀스 길이에 제한이 있음.
- 대부분의 모델은 최대 512개 또는 1024개의 토큰으로 구성된 시퀀스를 처리
- 더 긴 시퀀스를 처리해야 할 경우 충돌 발생
    - 지원되는 시퀀스 길이가 더 긴 모델 사용
    - 시퀀스를 잘라냄
"""
sequence = sequence[:max_sequence_length]

# ✏️ 직접 해 보세요!
# 2번 섹션의 두 문장을 토큰화하고 모델에 통과시켜 동일한 로짓을 얻는지 확인하기

print("\n=== 직접 해 보세요! 실습 ===\n")

# 1. 두 문장 개별적으로 토큰화 및 처리
sentence1 = "I've been waiting for a HuggingFace course my whole life."
sentence2 = "I hate this so much!"

# 각 문장을 개별적으로 토큰화
tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)

print(f"문장 1 토큰: {tokens1}")
print(f"문장 2 토큰: {tokens2}")
print(f"문장 1 토큰 길이: {len(tokens1)}")
print(f"문장 2 토큰 길이: {len(tokens2)}")

# 토큰을 ID로 변환
ids1 = tokenizer.convert_tokens_to_ids(tokens1)
ids2 = tokenizer.convert_tokens_to_ids(tokens2)

# 배치 차원 추가하여 텐서로 변환
input_ids1 = torch.tensor([ids1])
input_ids2 = torch.tensor([ids2])

# 각 문장을 개별적으로 모델에 통과시키기
outputs1 = model(input_ids1)
outputs2 = model(input_ids2)

print("\n개별 처리 결과:")
print(f"문장 1 로짓: {outputs1.logits}")
print(f"문장 2 로짓: {outputs2.logits}")

# 2. 패딩을 사용하여 두 문장을 배치로 처리 (어텐션 마스크 없이)
# 더 긴 문장에 맞춰 패딩 추가
max_length = max(len(ids1), len(ids2))
ids2_padded = ids2 + [tokenizer.pad_token_id] * (max_length - len(ids2))

batched_ids_no_mask = [ids1, ids2_padded]

outputs_batch_no_mask = model(torch.tensor(batched_ids_no_mask))
print("\n배치 처리 결과 (어텐션 마스크 없이):")
print(f"배치 로짓: {outputs_batch_no_mask.logits}")
print("주의: 패딩된 문장(두 번째)의 로짓이 개별 처리와 다름")

# 3. 어텐션 마스크를 사용하여 배치 처리
attention_mask = [
    [1] * len(ids1),  # 문장 1: 모든 토큰에 어텐션
    [1] * len(ids2) + [0] * (max_length - len(ids2))  # 문장 2: 패딩은 0으로 마스킹
]

outputs_batch_with_mask = model(
    torch.tensor(batched_ids_no_mask),
    attention_mask=torch.tensor(attention_mask)
)

print("\n배치 처리 결과 (어텐션 마스크 사용):")
print(f"배치 로짓: {outputs_batch_with_mask.logits}")
print("✅ 어텐션 마스크를 사용하면 개별 처리와 동일한 결과를 얻을 수 있음!")

# 결과 비교
print("\n=== 결과 비교 ===")
print(f"문장 1 - 개별: {outputs1.logits[0].tolist()}")
print(f"문장 1 - 배치: {outputs_batch_with_mask.logits[0].tolist()}")
print(f"문장 2 - 개별: {outputs2.logits[0].tolist()}")
print(f"문장 2 - 배치: {outputs_batch_with_mask.logits[1].tolist()}")