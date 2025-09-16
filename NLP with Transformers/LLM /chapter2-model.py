# 2. model
# 2.1 transformer 만들기

from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-cased")

"""
- from_pretrained() : Hugging Face Hub에서 모델 데이터를 다운로드하고 캐시.
- 체크포인트 이름은 특정 모델 아키텍처와 가중치에 해당하는데, 이 경우 기본 아키텍처(12개 레이어, 768개 히든 사이즈, 12개 어텐션 헤드)와
    대소문자 구분이 있는 입력(즉, 대소문자 구분이 중요함)을 가진 BERT 모델
"""

from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")

# 2.2 로딩 및 저장

"""
- config.json : 모델 아키텍처를 구축하는 데 필요한 속성
- pytorch_model.safetensors : 모델의 모든 가중치 포함
"""
model.save_pretrained("directory_on_my_computer")


# 2.2 텍스트 인코딩 
"""
- transformer 모델은 입력값을 숫자로 변환하여 텍스트를 처리
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)

#{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

"""
input_ids : 토큰의 수치 표현
token_type_ids : 입력의 어느 부분이 문장 A이고 어느 부분이 문장 B인지 모델에 알려줌
attention_mask : 어떤 토큰에 주의를 기울여야 하고 어떤 토큰은 무시해야 하는지
입력 ID 를 디코딩하여 원본 텍스트를 되찾을 수 있음
"""

tokenizer.decode(encoded_input["input_ids"])

# "[CLS] Hello, I'm a single sentence! [SEP]"

"""
- 토큰화기가 모델에 필요한 특수 토큰인 [CLS]와 [SEP]를 추가함
- 특수 토큰은 모델이 이를 포함하여 사전 훈련된 경우에만 사용되며, 이 경우 토큰화기는 해당 모델이 이러한 토큰을 기대하므로 이를 추가해야 함. 
"""

encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)

#{'input_ids': [[101, 1731, 1132, 1128, 136, 102], [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102]], 
# 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
# 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

"""
- 여러 문장을 전달할 때 토큰화기는 각 사전 값에 대해 각 문장별로 리스트를 반환
- 또한 토큰화기가 PyTorch 텐서를 직접 반환하도록 요청
"""

encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)

"""
{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102],
         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
"""

"""
문제점
- 두 리스트의 길이가 같지 않음. 
    배열과 텐서는 직사각형 형태여야 하므로, 이 리스트들을 단순히 PyTorch 텐서로 변환 불가능. 토큰화기는 이를 위해 패딩 옵션 제공
"""

# padding inputs
# 토큰화기에 입력에 패딩을 추가하도록 요청하면, 가장 긴 문장보다 짧은 문장들에 특수 패딩 토큰을 추가하여 모든 문장의 길이를 동일하게 만듦.

encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)

"""
{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102,     0,     0,     0,     0],
         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
"""

"""
- 패딩 토큰들은 ID 0으로 인코딩된 입력 ID로 변환되었고, 어텐션 마스크 값 역시 0
- 이는 해당 패딩 토큰들이 모델에 의해 분석되어서는 안 되기 때문
- 이들은 실제 문장의 일부가 아님
"""

# truncation inputs

"""
- 텐서가 모델이 처리할 수 있는 크기보다 커질 수 있음
- 예를 들어, BERT는 최대 512개의 토큰으로 구성된 시퀀스만 사전 훈련되었으므로 더 긴 시퀀스는 처리할 수 없음
- 모델이 처리할 수 있는 길이보다 긴 시퀀스가 있다면, 잘라내기 매개변수를 사용하여 시퀀스를 잘라내야 함.
"""

encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])

# padding, truncation 옵션 사용

encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)

"""
{'input_ids': tensor([[  101,  1731,  1132,  1128,   102],
         [  101,  1045,  1005,  1049,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]])}
"""

# adding a special tokens

encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])

# why is all of this necessary ?

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

"""
encoded_sequences = [
    [
        101,
        1045,
        1005,
        2310,
        2042,
        3403,
        2005,
        1037,
        17662,
        12172,
        2607,
        2026,
        2878,
        2166,
        1012,
        102,
    ],
    [101, 1045, 5223, 2023, 2061, 2172, 999, 102],
]
"""

import torch
model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)

"""
- 모델에서 텐서를 활용하는 방법 : 입력값과 함께 모델 호출
- 모델은 다양한 인수를 받아들이지만, 입력 ID 만 필수
"""