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
"""
- TensorDataset : 여러 텐서를 묶어서 하나의 데이터셋 객체로 만듦.
  여기서는 입력(input_ids), 마스크(attention_masks), 정답(labels)를 묶음.
  즉, train_data[i]를 꺼내면 (input_ids[i], mask[i], label[i]) 가 같이 나옴.

- RandomSampler(train_data) : 매 epoch 마다 훈련 데이터를 랜덤 순서로 섞음 -> 학습이 특정 순서에 치우치지 않도록 함.
- SequentialSampler(validataion_data) : 검증 데이터는 평가용이므로 순서대로 불러옴. 모델 성능을 확인할 목적이라 shuffle 필요 없음.

- DataLoader : 학습 시 데이터를 작은 배치 단위로 잘라서 모델에 공급해줌.
                전체 데이터를 한 번에 메모리에 안 올려도 됨(메모리 절약)
                for batch in train_dataloader : 형태로 반복문에서 간단히 사용 가능
                배치 단위로 GPU에 올려 연산 속도 최적화
"""
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler = train_sampler, batch_size= batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader=  DataLoader(validation_data, sampler = validation_sampler, batch_size = batch_size)

# 3.2.13 BERT 모델 설정하기
# BERT bert-base-uncased 설정 모델을 초기화하기
"""
1. Bert 기본 설정(Config)
- BertConfig : Bert 모델의 하이퍼파라미터(구조 설정값)을 담는 객체
  예 : hidden size, layer 수, attention head 수, dropout 확률 등
- BertConfig()만 호출하면 기본값()으로 설정됨
- BertModel은 순수한 BERT 인코더만 불러옴. (즉, Transforer만 있고 분류기 없음)
- 이 상태의 모델은 사전학습(MLM, NSP)에만 쓰거나, 위에 별도의 head(분류기 등)를 얹어야 downstream task에 활용 가능.

2. 모델 설정 확인
- configuration  = model.config
- 현재 모델이 가진 설정값

3. 사전학습된 BERT + 분류기 head 불러오기
- BertForSequenceClassification : 
  - BertModel 위에 분류기(fully connected layer)를 덧붙인 모델 
  - 문장 분류(이진 분류, 다중 분류)에 바로 사용 가능

- "bert-base-uncased" :
  - HuggingFace에서 제공하는 사전학습된 BERT checkpoint 로드
  - uncased 라서 대소문자 구분 안 함.

- num_labels = 2 :
  - 이진 분류 문제 (예 : 긍정/부정)
  - 출력층 크기가 2로 세팅됨

4. 다중 GPU 사용 (선택 사항)
- model = nn.DataParallel(model)
- 여러 GPU가 있을 경우 병렬 분산 학습 가능

5. 모델을 device(GPU/CPU) 로 이동
- model.to(device)
- 모델 파라미터와 연산을 GPU/CPU 메모리에 로드
- 이후 input_ids 같은 데이터도 .to(device)로 옮겨줘야 함.
"""
from transformers import BertModel, BertConfig
configuration = BertConfig()

# bert-base-uncased-style 설정을 사용해 모델 초기화하기
model = BertModel(configuration)

# 모델 설정 불러오기
configuration = model.config
print(configuration)

# 3.2.14 대소문자가 구분되지 않은 허깅페이스 BERT 모델 불러오기
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = nn.DataParallel(model)
model.to(device)

# 3.2.15 파라미터 그룹 옵티마이저
"""
- 모델 파라미터 옵티마이저 초기화
- 모델 미세 조정은 사전 학습된 모델 파라미터 값을 초기화하는 것으로 시작
- 파라미터 옵티마이저는 과적합을 방지하기 위한 가중치 감소율을 포함하고 일부 파라미터는 제외
"""

param_optimizer = list(model.named_parameters())

# 'weight' 파라미터를 'bias' 파라미터와 분리
no_decay = ['bias', 'LayerNorm.weight']

"""
- 'weight' 파라미터에 대해 'weight_decay_rate' 를 0.01로 설정
- 'bias' 파라미터에 대해 'weight_decay_rate' 를 0.0으로 설정
- 'bias'를 포함하지 않은 파라미터 필터링
- 'bias'를 포함한 파라미터 필터링
"""
optimizer_grouped_parameters = [
  {'params' : [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate' : 0.1},
  {'params' : [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate' : 0.0  }
  
]

# Displaying a sample of the parameter_optimizer:  layer 3
layer_parameters = [p for n, p in model.named_parameters() if 'layer.3' in n]

no_decay

# Displaying the list of the two dictionaries
small_sample = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)][:2],
     'weight_decay_rate': 0.1},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)][:2],
     'weight_decay_rate': 0.0}
]

for i, group in enumerate(small_sample):
    print(f"Group {i+1}:")
    print(f"Weight decay rate: {group['weight_decay_rate']}")
    for j, param in enumerate(group['params']):
        print(f"Parameter {j+1}: {param}")

# 3.2.16 학습 루프의 하이퍼파라미터
"""
- 학습률(learning rate, lr) 과 워밍업 비율(warmup)은 최적화 단계 초기에 매우 작은 값으로 설정하고 일정 횟수를 반복한 후에 점차적으로 증가해야 함.
- 이렇게 함으로써 경사가 커지고 최적화 목표를 지나치는 것을 방지 가능
"""

"""
1. 하이퍼파라미터 미 옵티마이저/스케줄러 설정
- epochs = 4 -> 학습 데이터셋을 4번 반복
- AdamW -> Adam의 변형 옵티마이저로, weight deay(정규화) 처리가 개선된 버전
          - optimizer_grouped_parameters : 이전에 만든 파라미터 그룹(weight decay 적용/미적용 등)을 넘겨 정규화를 다르게 줄 수 있음
          - lr = 2e-5 : BERT 파인튜닝에서 많이 쓰는 작은 학습률
          - eps = 1e-8 : 분모가 0에 가까워질 때의 수치 안정성을 위한 작은 실수

- total_steps -> 전체 학습 step 수 = (한 epoch의 step 수 x epoch 수)
- scheduler -> 학습 중 learning rate를 step 마다 조금씩 줄여주는 스케줄러
               여기서는 선형 감소 (linear decay) 스케줄을 사용
"""
# optimizer = BertAdam(optimizer_grouped_parameters,
#                      lr=2e-5,
#                      warmup=.1)

epochs = 4

optimizer = AdamW(optimizer_grouped_parameters,lr = 2e-5, eps = 1e-8)

total_steps =  len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


"""
2. 정확도 함수
- 모델 출력 pred(로짓 또는 확률)에서 axis = 1로 클래스 argmax 를 뽑아 예측 라벨로 만듦.
- flatten()은 배치 차원을 평탄화해 비교를 쉽게 만듦.
- 마지막 줄에서 예측과 정답이 같은 개수를 세어 전체 길이로 나눔.
"""
def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis = 1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 3.2.17 학습 루프

# training loop
t = [] 

# 스텝마다 loss 기록 저장
train_loss_set = []

# tqdm의 trange -> 진행률 바를 보여주는 편의 기능
for _ in trange(epochs, desc = "Epoch"):

  # 학습 모드로 전환. Dropout 등이 활성화 됨.
  model.train()

  # 에폭 평균 loss 계산을 위해 누적변수 초기화
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0

  # 데이터로더에서 배치를 하나씩 꺼냄
  for step, batch in enumerate(train_dataloader):
    # 배치의 모든 텐서를 GPU/CPU로 이동
    batch = tuple(t.to(device) for t in batch)
    # 입력 토큰, attention mask, 정답 label을 unpack.
    b_input_ids, b_input_mask, b_labels = batch
    # gradient 초기화 (pytorch는 기본이 누적이므로, 매 스텝 초기화 필요)
    optimizer.zero_grad()
    # 순전파. labels 를 넘기면 hugging face 모델이 cross-entropy loss 를 내부에서 계산해 함께 반환.
    # 일반적으로 outputs에는 loss, logits, hidden_states 등이 들어있는 딕셔너리가 옴.
    outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
    # 이번 스텝의 스칼라 loss를 뽑아 기록
    loss = outputs['loss']
    train_loss_set.append(loss.item())
    # 역전파로 파라미터별 그래디언트 계산
    loss.backward()
    # 가중치 업데이트
    optimizer.step()

    # 학습률 스케줄 업데이트
    scheduler.step()

    # 평균 loss를 계산하기 위해 스텝별 loss 수와 샘플 수를 누적
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss : {}".format(tr_loss / nb_tr_steps))

  # Validation

  # 평가모드 전환
  # 드롭아웃/배치정규화가 평가 모드로 바뀜.(학습 시의 불확실성 / 노이즈를 끄고, 일관된 추론을 하게 함.)
  model.eval()

  # 지표 누적 변수 초기화
  # 정확도/스텝 수를 누적하려고 만듦
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0
 
  # 검증 배치 반복 : 검증 데이터 한 배치씩 꺼내서 디바이스로 보냄.
  for batch in validation_dataloader:
    # add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # unpack the inputs from our dataloader
    b_input_ids , b_input_mask, b_labels = batch
    # 추론만 실시. (그래디언트 계산 없음.)
    # torch.no_grad()로 메모리/속도 절약(그래디언트, 중간 값 저장 X).
    # 검증에서는 보통 로짓/확률만 필요하므로 labels 를 안 넣음 -> 모델이 loss를 계산하지 않음.
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask =b_input_mask)

    # CPU로 옮겨 정확도 계산
    # 넘파이로 바꿔 flat_accuracy(argmax 기반 단일 라벨 정확도) 계산.
    # 배치별 정확도를 스텝 평균으로 누적.
    logits = logits['logits'].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  print("Validation Accuracy : {}".format(eval_accuracy / nb_eval_steps))

# 3.2.18 학습 평가하기

plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()

# 3.2.19 홀드아웃 데이터셋을 사용하여 예측 및 평가하기
# 1. 데이터 로드
# 탭 구분 TSV 를 읽어와 4개 컬럼으로 이름 부여.
df = pd.read_csv("out_of_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

print(df.shape)

# 2. 문장/라벨 분리
# 모델 입력용 문장 배열과 평가용 정답 라벨을 뽑음
sentences = df.sentence.values
labels = df.label.values

# 3. 토큰 추가
# BERT 입력 형식인 [CLS]...[SEP]를 직접 붙인 다음 토크나이즈
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# 4. 정수 ID로 변환 + 패딩/트렁케이션
# 토큰을 vocab 인덱스로 바꾸고, 길이를 최대 128로 맞춤
# 뒤쪽 자르기(post) + 뒤쪽 패딩(post)
MAX_LEN = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen = MAX_LEN, dtype="long", truncating="post", padding="post")

# 5. 어텐션 마스크 생성
# 0(패딩) 위치는 0, 실제 토큰 위치는 1로 마스크 생성(모델이 패딩을 무시하도록 함.)
attention_masks = []
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

# 6. 텐서화
# 추론/평가에 쓸 입력/마스크/라벨을 텐서로 변환
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# 7. DataLoader(순차 샘플러 SequentialSampler)
batch_size = 32
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler = prediction_sampler, batch_size = batch_size)

"""
| 항목      | 학습(Training)                      | 검증(Validation)               | 홀드아웃(Prediction/Eval)        |
| ------- | --------------------------------- | ---------------------------- | ---------------------------- |
| 데이터 용도  | 파라미터 업데이트                         | 하이퍼파라미터·과적합 확인               | **완전 미사용 데이터로 최종 일반화 성능 확인** |
| 샘플러     | **RandomSampler(셔플)** 권장          | 보통 `SequentialSampler` 또는 고정 | **SequentialSampler** 사용     |
| 라벨 사용   | `labels`로 **loss 계산 + backward**  | 보통 `labels`로 **지표/val loss** | **지표 계산**(정확도 등)만            |
| 그라디언트   | `backward()` + `optimizer.step()` | 없음 (`no_grad`)               | 없음 (`no_grad`)               |
| 스케줄러/LR | `scheduler.step()`                | 없음                           | 없음                           |
| 토큰화/전처리 | 동일                                | 동일                           | **동일(단, 보통 셔플 X)**           |

"""

# 1. 소프트맥스 함수 정의 : 넘파이로 직접 만든 소프트맥스
def softmax(logits):
  e = np.exp(logits)
  return e / np.sum(e)

# 2. 평가 모드 전환
# 평가 모드 : dropout, batchnorm 등의 학습 특성을 꺼서 예측이 안정적으로 나오도록 함.
model.eval()

# 3. 추적용 리스트 초기화
# 원시 로짓, 예측 클래스, 실제 라벨을 저장할 리스트 
raw_predictions, predicted_classes, true_labels = [], [] , []

# 4. 배치 반복
# prediction_dataloader에서 배치 단위로 데이터 꺼내옴.
# device(GPU/CPU)로 옮기고 언팩.
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  # 5. 순전파(추론)
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask)

  # 6. 로짓과 라벨을 CPU로 이동
  logits = outputs['logits'].detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  # 7. 입력 토큰을 다시 문장으로 디코딩
  # 토큰 인덱스를 원래 문장 텍스트로 복원
  # skip_special_tokens = True 로 [CLS], [SEP], [PAD] 등을 제거
  b_input_ids = b_input_ids.to('cpu').numpy()
  batch_sentences = [tokenizer.decode(input_ids, skip_special_tokens = True) for input_ids in b_input_ids]

  # 8. 소프트맥스 확률 계산
  # logits -> 확률 변환
  # dim = 1 은 각 샘플의 클래스 차원에 대해 softmax
  probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim =1)

  # 9. 예측 클래스
  batch_predictions = np.argmax(probabilities, axis =1)


  # 10. 결과 출력
  for i, sentence in enumerate(batch_sentences):
    print(f"Sentence: {sentence}")
    print(f"Prediction: {logits[i]}")
    print(f"Sofmax probabilities", softmax(logits[i]))
    print(f"Prediction: {batch_predictions[i]}")
    print(f"True label: {label_ids[i]}")

  # 11. 결과 저장
  raw_predictions.append(logits)
  predicted_classes.append(batch_predictions)
  true_labels.append(label_ids)

# 3.2.20 매튜 상관 계수를 사용하여 평가하기
"""
Matthews correlation coefficient (MCC) 란?
- 이진 분류에서 정확도보다 더 균형 잡힌 지표
- 양성/음성 데이터가 불균형할 때 특히 유용.
- 값 범위 :
    - +1 : 완벽한 예측
    - 0 : 랜덤 예측 수준
    - -1 : 완전히 반대 예측
  즉, 데이터셋이 불균형일 때 정확도보다 모델 성능을 잘 반영
"""

from sklearn.metrics import matthews_corrcoef

# 3.2.21 개별 배치의 점수

# 배치별 MCC 점수를 담을 리스트 초기화
matthews_set = []

# true_labels와 predicted_classes 는 앞서 예측 루프에서 만든 배치 단위 결과
# 따라서 배치 개수만큼 반복
for i in range(len(true_labels)):
  # Calculate the Matthews correlation coefficient for this batch

  # true_labels[i] are the true labels for this batch
  # predicted_classes[i] are the predicted classes for this batch
  # We don't need to use np.argmax because predicted_classes already contains the predicted classes

  # 각 배치의 정답 라벨 배열 vs 예측 라벨 배열을 비교해 MCC 계산
  # predicted_classes[i]는 이미 argmax 결과이므로 그대로 사용
  matthews = matthews_corrcoef(true_labels[i], predicted_classes[i])

  # 배치별 MCC 점수를 리스트에 추가
  matthews_set.append(matthews)


# 3.2.22 전체 데이터셋의 매튜 평가
# 원래 true_labels / predicted_classes 는 배치별 리스트 (예 [[0,1,1], [1,0,1], ...]) 형태
# 이중 리스트를 평탄화(flatten) 해서 단일 리스트로 만듦 -> true_labels_flattened = [0,1,1,1,0,1,...]
true_labels_flattened = [label for batch in true_labels for label in batch]
predicted_classes_flattened = [pred for batch in predicted_classes for pred in batch]

# 사이킷런의 matthews_corrcoef 로 전체 데이터 기준 MCC 계산
# 배치별 MCC 평균보다 더 정확한 "전역 지표"
mcc = matthews_corrcoef(true_labels_flattened, predicted_classes_flattened)

# 최종 MCC 점수 출력
print(f"MCC: {mcc}")

# 3.3 정리하기
"""
- BERT 는 트랜스포머에 양방향 어텐션 도입
- BERT 는 두 단계 프레임워크로 설계
  - 모델을 사전 학습
  - 모델을 미세 조정
"""