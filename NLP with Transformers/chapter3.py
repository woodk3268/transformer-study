# chapter 3. BERT fine-tuning

# 3.1 BERT architecture
"""
- BERT는 트랜스포머 모델에 양방향 어텐션 적용
- 기존 트랜스포머 모델에 양방향 어텐션을 적용하려면 많은 변경 필요
"""

# 3.1.1 인코더 스택
"""
1. 오리지널 트랜스포머

* 인코더와 디코더 모두 사용.
* 인코더/디코더 층 수: **N=6**
* 은닉 차원: **d_model = 512**
* 멀티헤드 어텐션 헤드 수: **A=8**
* 각 헤드 차원: **d_k = 512 / 8 = 64**

2. BERT 특징

* **디코더 없음** → 인코더만 사용.
* 마스크드 토큰 예측(MLM)을 위해 인코더의 self-attention에 마스크를 적용.
* 오리지널 트랜스포머보다 더 크고 깊은 인코더 스택을 사용.

3. BERT_BASE

* 층 수: **N=12**
* 은닉 차원: **d_model = 768 (H=768)**
* 어텐션 헤드 수: **A=12**
* 각 헤드 차원: **d_k = 768 / 12 = 64**
* 출력: 12개 헤드의 결과를 연결(concatenate)한 벡터.

4. BERT_LARGE

* 층 수: **N=24**
* 은닉 차원: **d_model = 1024**
* 어텐션 헤드 수: **A=16**
* 각 헤드 차원: **d_k = 1024 / 16 = 64**
* 출력: 16개 헤드의 결과를 연결한 벡터.

5. 요약 

* **차원 수(H, d_model)와 층 수(N)** 가 크면 더 많은 정보를 담을 수 있음.
* **헤드 개수 증가**는 다양한 표현(subspace)을 병렬적으로 학습할 수 있게 해줌.
* 결국 BERT는 **더 큰 모델(더 많은 메모리) + 더 많은 데이터 사전 학습** → 다양한 NLP 다운스트림 작업에서 뛰어난 성능을 달성.


"""

# 3.1.2 사전 학습 입력 환경 준비하기

# 3.1.2.1 마스크드 언어 모델링(Masked Language Modeling, MLM)
"""
- BERT는 무작위로 문장의 한 단어에 마스크를 씌운 후 양방향으로 분석하는 방식 사용
- 입력에 단어 분할(subword segmentation) 토큰화 방법인 워드피스(WordPiece)를 적용
- 학습된 위치 인코딩(learned positional encoding) 사용
"""

# 3.1.2.2 다음 문장 예측하기(Next Sentence Prediction, NSP)
"""
1. 다음 문장 예측 (Next Sentence Prediction, NSP)

* 입력: **두 개의 문장**

  * 절반은 실제 연속된 문장 쌍 (**양성 샘플, IsNext**).
  * 절반은 임의로 선택된 문장 쌍 (**음성 샘플, NotNext**).

2. 새로운 특수 토큰

* **[CLS]**

  * 입력 시퀀스 맨 앞에 추가.
  * 전체 시퀀스의 표현을 담아 이진 분류(NSP) 등에 사용.
  * 두 번째 시퀀스가 첫 번째 시퀀스와 연속되는지 예측하기 위해 사용
* **[SEP]**

  * 시퀀스(문장)의 끝을 표시하는 분리 토큰.
  * 문장 A와 B를 구분할 때도 사용.

3. 입력 임베딩 구성 요소

최종 입력 임베딩 =

* **토큰 임베딩** (각 단어/서브워드)
* **세그먼트(문장) 임베딩** (문장 A, 문장 B 구분)
* **위치 임베딩** (순서 정보 반영)

이 세 가지를 합(sum)하여 사용.

4. 추가 특징
- 양방향 어텐션을 사용
- 라벨링되지 않은 텍스트를 비지도 방식으로 사전 학습.
- 사전 학습 과정의 모든 단계를 포괄하는 지도 학습도 사용
"""

# 3.1.3 BERT 모델 사전 학습 및 미세조정
"""
1) 사전 학습
- 모델 아키텍처 정의 : 층 수, 헤드 수, 차원 및 기타 모델의 구성 요소 정의
- MLM 및 NSP로 모델 학습하기

2) 미세 조정
- 사전 학습된 BERT 모델의 파라미터로 선택한 다운스트림 모델 초기화하기
- 특정 다운스트림 작업을 위해 파라미터 미세조정하기
"""

# 3.2 BERT 미세 조정하기

# 3.2.1 하드웨어 제약사항

# 3.2.2 BERT를 위한 허깅페이스 파이토치 인터페이스 설치하기
# pip install -q transformers

# 3.2.3 모듈 불러오기

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange  #for progress bars
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image #for image rendering

# 3.2.4 토치용 장치로 CUDA 지정하기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3.2.5 데이터셋 불러오기

import os

os.system("curl -L https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-2nd-Edition/master/Chapter03/in_domain_train.tsv --output in_domain_train.tsv")

os.system("curl -L https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-2nd-Edition/master/Chapter03/out_of_domain_dev.tsv --output out_of_domain_dev.tsv")


#source of dataset : https://nyu-mll.github.io/CoLA/
df = pd.read_csv("in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
df.shape

df.sample(10)

# 3.2.6 문장, 라벨 목록 만들기 및 BERT 토큰 추가하기
#@ 문장, 라벨 리스트 생성하기 및 BERT 토큰 추가하기
sentences = df.sentence.values

#BERT에 사용하기 위해 CLS, SEP 토큰을 각 문장의 시작과 끝에 추가하기
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

# 3.2.7 BERT 토크나이저 활성화하기
"""
- 사전 학습된 BERT 토크나이저 초기화
"""
#@제목 BERT 토크나이저 활성화하기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print("Tokenize the first sentence")
print(tokenized_texts[0])

# 3.2.8 데이터 처리하기
# 최대 시퀀스 길이 설정
# 논문에서 저자는 512 길이를 사용했었다.
MAX_LEN =128

# 토큰 -> 인덱스 변환
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 패딩과 시퀀스 길이 맞추기
input_ids = pad_sequences(input_ids, maxlen = MAX_LEN, dtype="long", truncating = "post", padding="post")


# 3.2.9 어텐션 마스크 만들기
attention_masks = []

# 각 토큰에 대해 1 마스크를 생성하고 패딩에 대해 0 마스크 생성
"""
input_ids 에는 이미 [토큰ID, ..., 0, 0, 0] 형태로 패딩이 포함되어 있음.
i > 0 이면 실제 토큰 -> 1.0
i == 0 이면 패딩 토큰 -> 0.0
즉, attention_masks 는 실제 단어 위치만 1, 패딩은 0으로 마킹하는 벡터
"""
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

# 3.2.10 데이터를 학습 및 검증 셋으로 분할하기
# train_test_split을 사용해 데이터를 학습 및 검증 데이터셋으로 분리
train_inputs,validation_inputs,train_labels, validation_labels = train_test_split(input_ids, labels, random_state = 2018, test_size = 0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state = 2018, test_size = 0.1)

# 3.2.11 모든 데이터를 토치 텐서로 변환하기
# 모델에 데이터를 입력하기 위해 토치 텐서 타입으로 변환해야 함
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# 3.2.12 배치 크기 선택 및 이터레이터 만들기
# 학습에 사용될 배치 사이즈 선택. BERT를 특정 테스크를 위해 미세 조정하기 위해서는 16 또는 32의 배치 사이즈 추천
batch_size = 32

# 토치 DataLoader 를 사용해 데이터 이터레이터 생성. 이렇게 하면 학습 과정에서 루프를 사용하는 것보다 메모리 사용을 줄일 수 있다.
# iterator를 사용하면 전체 데이터를 메모리에 한번에 로드할 필요 없다.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler = train_sampler, batch_size= batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader=  DataLoader(validation_data, sampler = validation_sampler, batch_size = batch_size)

# 3.2.13 BERT 모델 설정하기
# BERT bert-base-uncased 설정 모델을 초기화하기

from transformers import BertModel, BertConfig
configuration = BertConfig()

# bert-base-uncased-style 설정을 사용해 모델 초기화하기
model = BertModel(configuration)

# 모델 설정 불러오기
configuration = model.configuration
print(configuration)

# 3.2.14 대소문자가 구분되지 않은 허깅페이스 BERT 모델 불러오기
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = nn.DataParallel(model)
model.to(device)