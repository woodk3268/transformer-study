# 1. 데이터 처리

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1.1 허브에서 데이터 세트 로드

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

"""
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
"""

"""
- 훈련세트, 검증 세트, 테스트 세트를 포함하는 DatasetDict 객체
- 각 데이터 세트는 여러 열(sentence1, sentence2, label, idx)과 가변적인 행 수를 포함하며, 행 수는 각 세트의 요소 수를 나타냄
"""

# 사전과 마찬가지로 인덱싱을 통해 객체의 각 문장 쌍에 접근 가능

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

"""
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
"""

print(raw_train_dataset.features)

"""
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
"""

"""
- 배경에서 label은 ClassLabel 유형이며, 정수와 레이블 이름 간의 매핑은 names 폴더에 저장됨.
- 0은 not_equivalent에, 1은 equivalent에 해당함.
"""

# 1.2 데이터 세트 전처리 
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 개별 문장 토큰화 (첫 번째 샘플만)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"][0])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"][0])

print("First sentence tokenized:")
print(tokenized_sentences_1)
print("\nSecond sentence tokenized:")
print(tokenized_sentences_2)

# 또는 배치로 처리 (처음 5개)
batch_tokenized_1 = tokenizer(raw_datasets["train"]["sentence1"][:5])
print(f"\nBatch tokenized (5 samples): {len(batch_tokenized_1['input_ids'])} sequences")

"""
- 데이터셋을 전처리하기 위해 텍스트를 모델이 이해할 수 있는 숫자로 변환해야 함.
- 토크나이저를 통해 수행됨.
- 토크나이저에 한 문장 또는 문장 목록을 입력할 수 있으므로 각 쌍의 첫 번째 문장과 두 번째 문장을 다음과 같이 직접 토큰화
"""

# 토크나이저는 시퀀스 쌍을 받아서 BERT 모델이 예상하는 방식으로 처리
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)

"""
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
"""

"""
- 단순히 두 시퀀스를 모델에 전달해서는 예측을 얻을 수 없음.
- 두 시퀀스를 한 쌍으로 처리하고 적절한 전처리 과정을 적용해야 함.
"""

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

"""
● Token Type IDs 정리

  기본 구조

  토큰:     [CLS] this is first . [SEP] this is second . [SEP]
  타입ID:     0    0   0    0   0   0     1   1     1   1   1
              └─── 첫 번째 문장 ───┘     └─── 두 번째 문장 ──┘

"""

# ========== 시퀀스 쌍 토큰화: 좋은 예 vs 나쁜 예 ==========

print("\n" + "="*60)
print("시퀀스 쌍 토큰화 예제: 좋은 예 vs 나쁜 예")
print("="*60)

# ❌ 나쁜 예 1: 문장을 따로 토큰화
print("\n❌ 나쁜 예 1 - 문장 개별 토큰화:")
bad_tok1 = tokenizer(raw_datasets["train"]["sentence1"][0])
bad_tok2 = tokenizer(raw_datasets["train"]["sentence2"][0])
print(f"sentence1 token_type_ids: {bad_tok1['token_type_ids'][:10]}...")
print(f"sentence2 token_type_ids: {bad_tok2['token_type_ids'][:10]}...")
print("→ 문제: 두 문장이 연결되지 않고, token_type_ids가 모두 0")
print("→ 모델은 두 문장의 관계를 파악할 수 없음")

# ❌ 나쁜 예 2: 전체 데이터셋을 한 번에 토큰화 (메모리 문제)
print("\n❌ 나쁜 예 2 - 전체 데이터셋 한 번에 토큰화 (주의: 실제로는 실행하지 않음):")
print("# 이렇게 하면 메모리 문제 발생 가능:")
print("# tokenized_dataset = tokenizer(")
print("#   raw_datasets['train']['sentence1'],  # 3668개 모두")
print("#   raw_datasets['train']['sentence2'],  # 3668개 모두")
print("#   padding=True, truncation=True)")
print("→ 문제: RAM 사용량이 폭발적으로 증가")

# ✅ 좋은 예 1: 단일 쌍 올바른 토큰화
print("\n✅ 좋은 예 1 - 문장 쌍 함께 토큰화:")
good_single = tokenizer(
    raw_datasets["train"]["sentence1"][0],
    raw_datasets["train"]["sentence2"][0]
)
print(f"token_type_ids: {good_single['token_type_ids'][:25]}...")
print("→ 첫 번째 문장은 0, [SEP] 후 두 번째 문장은 1로 구분")
print(f"→ 구조: [CLS]=101 ... [SEP]=102 ... [SEP]=102")

# ✅ 좋은 예 2: 작은 배치로 처리
print("\n✅ 좋은 예 2 - 작은 배치 처리:")
small_batch = tokenizer(
    raw_datasets["train"]["sentence1"][:3],
    raw_datasets["train"]["sentence2"][:3],
    padding=True,
    truncation=True,
    max_length=50
)
print(f"배치 크기: {len(small_batch['input_ids'])} 샘플")
print(f"첫 샘플 shape: {len(small_batch['input_ids'][0])} 토큰")

# ✅ 좋은 예 3: Dataset.map()으로 효율적 처리
print("\n✅ 좋은 예 3 - Dataset.map() 사용 (가장 효율적):")
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,  # 배치 처리로 속도 향상
    batch_size=1000  # 한 번에 1000개씩 처리
)
print(f"처리된 데이터셋 키: {list(tokenized_datasets['train'].features.keys())}")
print("→ 메모리 효율적, 병렬 처리 가능")

print("\n" + "="*60)

"""
단점 (나쁜 예 2의 경우):
- 토큰화 과정에서 전체 데이터셋을 저장할 만큼 충분한 RAM이 있어야만 작동
- Datasets 라이브러리의 데이터셋은 요청한 샘플만 메모리에 로드됨
"""