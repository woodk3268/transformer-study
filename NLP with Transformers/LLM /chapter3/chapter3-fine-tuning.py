# 2. Fine-tuning a model with the Trainer API

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation = True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

# 2.1 Training
"""
- 우선, 훈련 및 평가에 사용할 모든 하이퍼파라미터를 포함할 TrainingArguments 클래스를 정의해야 함.
- 반드시 제공해야 하는 유일한 인수는 훈련된 모델과 중간 체크포인트가 저장될 디렉터리
- 나머지 모든 항목은 기본값을 그대로 두어도, 기본적인 파인튜닝에는 잘 작동
"""

from transformers import TrainingArguments, AutoModelForSequenceClassification

training_args = TrainingArguments("test-trainer")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

"""
- 모델을 생성하면 지금까지 생성된 모든 객체를 전달하여 Trainer 정의 가능
- processing_class : Trainer 처리에 사용할 토크나이저를 알려주는 기능
"""

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_dataset["validation"],
    data_collator = data_collator,
    processing_class = tokenizer
)

trainer.train()

# 2.2 Evaluation

"""
- compute_metrics()
- 객체(EvalPrediction로 구성된 튜플)를 받아야 하며, 문자열을 부동 소수점으로 매핑하는 사전을 반환
- 문자열은 반환되는 메트릭의 이름이고 부동 소수점(실수)은 그 값임. 
"""

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

"""
- predict() 메서드의 출력은 세 개의 필드(predictions, label_ids, metrics)를 가진 튜플
- metrics 필드에는 전달된 데이터셋에 대한 손실값과 일부 시간 메트릭이 포함됨
- predictions 는 408x2 크기의 2차원 배열(predict()에 전달한 데이터셋의 각 요소에 대한 로짓 값)
- 이를 레이블과 비교할 수 있는 예측값으로 변환하려면 두 번째 축에서 최댓값을 가지는 인덱스를 취해야 함.
"""

import numpy as NLP
preds =np.argmax(predictions.predictions, axis = -1)

import evaluate
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references= prediction.label_ids)
# {'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}

def compute_metrics(eval_preds):
    metric = evaluate.load("glue","mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(prediction = predictions, references= labels)

training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels =2)

trainer = Trainer(
    model,
    training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class = tokenizer,
    compute_metrics = compute_metrics,

)

trainer.train()

