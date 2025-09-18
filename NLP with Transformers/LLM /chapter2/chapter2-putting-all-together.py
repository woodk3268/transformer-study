from transformers import AutoTokenizer

  # AutoTokenizer가 내부적으로 하는 일:
  # 1. tokenizer_config.json에서 "tokenizer_class" 확인
  # 2. 해당 클래스를 자동으로 import하고 인스턴스 생성
  # 3. 저장된 vocabulary 파일(vocab.txt 등) 로드
  # 4. 특수 토큰 설정 적용

  # 실제로는 이렇게 동작:
  """
  if "distilbert" in model_type:
      return DistilBertTokenizer.from_pretrained(checkpoint)
  elif "bert" in model_type:
      return BertTokenizer.from_pretrained(checkpoint)
  elif "gpt2" in model_type:
      return GPT2Tokenizer.from_pretrained(checkpoint)
  """
  # ... 등등
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


  # model_inputs 변수는 모델 실행에 필요한 모든 입력값을 담고 있는 딕셔너리
  # 
  # DistilBERT 모델의 경우 필요한 입력값:
  #   - input_ids: 토큰화된 텍스트의 숫자 ID
  #   - attention_mask: 패딩 토큰을 무시하기 위한 마스크
  #
  # 다른 모델들의 경우 추가 입력이 필요할 수 있습니다:
  #   - BERT: input_ids, attention_mask, token_type_ids
  #   - GPT-2: input_ids, attention_mask, position_ids
  #   - RoBERTa: input_ids, attention_mask
  #
  # tokenizer()가 반환하는 model_inputs 객체는 
  # 해당 모델에 필요한 모든 입력을 자동으로 준비해줌.
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences) # 여러 시퀀스를 동시에 처리


model_inputs = tokenizer(sequences, padding="longest") # 시퀀스를 최대 시퀀스 길이까지 패딩


model_inputs = tokenizer(sequences, padding="max_length") # 시퀀스를 모델의 최대 길이까지 패딩


model_inputs = tokenizer(sequences, padding="max_length", max_length=8) # 시퀀스를 지정된 최대 길이까지 패딩


# truncate

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences, truncation=True) # 모델 최대 길이보다 긴 시퀀스를 잘라냄

model_inputs = tokenizer(sequences, max_length =8, truncation = True) # 지정된 최대 길이보다 긴 시퀀스를 잘라냄

# 객체 tokenizer는 특정 프레임워크 텐서로의 변환을 처리할 수 있으며, 변환된 텐서는 모델로 직접 전송될 수 있음.
# 예를 들어, 아래 예제에서는 토크나이저가 다양한 프레임워크의 텐서를 반환하도록 요청
# 즉, "pt" PyTorch 텐서와 "np" numpy 배열을 반환

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences, padding= True , return_tensors= "pt" )  # PyTorch 텐서를 반환 
model_inputs = tokenizer(sequences, padding= True , return_tensors= "np" ) # NumPy 배열을 반환

# special tokens

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))

#"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
#"i've been waiting for a huggingface course my whole life."

  # 토크나이저는 문장 앞뒤에 특수 토큰을 자동으로 추가
  #
  # BERT/DistilBERT의 특수 토큰:
  #   [CLS] - 문장 시작 표시 (Classification의 약자)
  #   [SEP] - 문장 끝 표시 (Separator의 약자)
  #   예: [CLS] Hello world [SEP]
  #
  # 왜 특수 토큰이 필요한가?
  #   - 모델이 학습할 때 이 토큰들과 함께 학습했음
  #   - [CLS]: 문장 전체의 의미를 담는 토큰 (분류 작업에 사용)
  #   - [SEP]: 문장의 경계를 구분하는 토큰
  #
  # 모델별 특수 토큰 차이:
  #   - BERT: [CLS] 문장 [SEP]
  #   - GPT-2: 특수 토큰 없음 (또는 <|endoftext|>)
  #   - RoBERTa: <s> 문장 </s>
  #   - T5: 특수 접두사 추가 (예: "translate: ")
  #
  # 핵심: 토크나이저가 자동으로 처리
  #   - 각 모델에 맞는 특수 토큰을 자동 추가
  #   - 개발자가 직접 추가할 필요 없음
  #   - from_pretrained()로 불러오면 모델과 일치하는 설정 적용

# 마무리

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation = True, return_tensors="pt")
output = model(**tokens)