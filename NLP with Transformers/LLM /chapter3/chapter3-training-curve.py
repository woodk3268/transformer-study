# 4. 학습 곡선 이해
# 4.1 학습 곡선이란 무엇인가?
"""
- 학습 곡선은 학습 중 시간 경과에 따른 모델의 성능 지표를 시각적으로 표현한 것
    - 손실 곡선 : 모델의 오류(손실)가 학습 단계 또는 에포크에 따라 어떻게 변하는지 보여줌
    - 정확도 곡선 : 훈련 단계 또는 에포크에 대한 정확한 예측의 백분율을 표시
"""

# 4.1.1 손실 곡선
"""
- 높은 초기 손실 : 모델이 최적화 없이 시작되므로 예측이 처음에는 좋지 않음.
- 손실 감소 : 훈련이 진행됨에 따라 손실은 일반적으로 감소해야함.
- 수렴 : 결국 손실은 낮은 값으로 안정화되어 모델이 데이터의 패턴을 학습했음을 나타냄
"""

from transformers import Trainer, TrainingArguments
import wandb

wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "steps",
    eval_steps = 50,
    save_steps = 100,
    logging_step = 10, # log metrics every 10 steps
    num_train_epochs = 3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size = 16,
    report_to= "wandb" # send logs to Weights & Biases
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    data_collator = data_collator,
    processing_class = tokenizer,
    compute_metrics = compute_metrics,
)

# Train and automatically log metrics
trainer.train()

# 4.1.2 정확도 곡선
"""
- 시간에 따른 올바른 예측 비율을 보여줌
- 손실 곡선과 달리, 일반적으로 모델이 학습됨에 따라 증가해야 함.

- 낮게 시작 : 초기 정확도는 낮아야 함. 모델이 아직 데이터의 패턴을 학습하지 않았음
- 훈련에 따라 증가 : 모델이 데이터의 패턴을 학습할 수 있다면, 일반적으로 모델이 학습함에 따라 정확도가 향상되어야 함.
- 정체 현상 발생 가능 : 모델이 실제 레이블에 근접한 예측을 할 때 정확도는 부드럽게 증가하기보다 불연속적인 점프 형태로 증가하는 경우가 많음
"""

# 4.1.3 수렴
"""
- 수렴은 모델의 성능이 안정화되고 손실 및 정확도 곡선이 평탄해질 때 발생
- 이는 모델이 데이터의 패턴을 학습하여 사용 준비가 되었음을 나타냄
- 모델이 수렴한 후에는 이를 활용해 새로운 데이터에 대한 예측을 수행하고, 평가 지표를 참조하여 모델의 성능 파악 가능
"""

# 4.1.4 학습 곡선 패턴 해석

"""
- Healthy Learning Curves
    - 손실의 부드러운 감소 : 훈련 손실과 검증 손실 모두 꾸준히 감소
    - 훈련/검증 성능의 근접성 : 훈련 지표와 검증 지표 간 차이
    - 수렴성 : 곡선이 평탄해져 모델이 패턴을 학습했음을 나타냄

- Overfitting
    - 과적합은 모델이 훈련 데이터로부터 지나치게 학습하여 검증 데이터(검증 세트)로 일반화하지 못하는 형상
    - 증상 : 
        - 훈련 손실은 계속 감소하는 반면 검증 손실은 증가하거나 정체됨
        - 훈련 정확도와 검증 정확도 간 큰 차이
        - 훈련 정확도가 검증 정확도보다 훨씬 높음

- Underfitting
    - 모델이 데이터의 근본적인 패턴을 포착하기에는 너무 단순할 때 발생
    - 증상 :
        - 훈련 손실과 검증 손실 모두 높은 상태 유지
        - 모델 성능이 훈련 초기에 정체됨
        - 훈련 정확도가 예상보다 낮음

- Erratic Learning Curves
    - 모델이 효과적으로 학습하지 못할 때 불규칙한 학습 곡선이 발생
    - 증상 :
        - 손실 또는 정확도에서 빈번한 변동 발생
        - 곡선이 높은 분산 또는 불안정성을 보임
        - 성능이 명확한 추세 없이 진동함
        - 훈련 및 검증 곡선 모두 불규칙한 행동을 보임

        
"""
from transfromers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "steps",
    eval_steps = 100,
    save_strategy = "steps",
    save_steps = 100,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    greater_is_better = False,
    num_train_epochs =  10 # Set high, but we'll stop early
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    data_collator = data_collator,
    processing_class = tokenizer,
    compute_metrics = compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    -num_train_epochs=5,
    +num_train_epochs=10,
)

training_args = TrainingArguments(
    output_dir = "./results",
    -learning_rate = 1e-5,
    +learning_rate = 1e-4,
    -per_device_train_batch_size = 16,
    +per_device_train_batch_size = 32,
)

