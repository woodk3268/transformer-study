# 4. RoBERTa 모델 처음부터 사전 학습하기
# 4.1 토크나이저 학습하기 및 트랜스포머 사전 학습하기

"""
- KantaiBERT는 BERT의 아키텍처를 기반으로 하는 RoBERTa 와 유사한 모델
- KantaiBERT는 6개의 층, 12개의 헤드, 84,095,008 개의 파라미터로 구성된 작은 모델
- 바이트 수준의 바이트 쌍 인코딩 토크나이저 구현
"""

# 4.2 처음부터 KantaiBERT 구축하기
# 4.2.1 1단계 : 데이터셋로드하기

# 4.2.2 2단계 : 허깅페이스 트랜스포머 설치하기

# 4.2.3 3단계 : 토크나이저 학습하기

"""
토크나이저 : 텍스트 -> 토큰(숫자 ID) 으로 바꿔주는 도구
"어떤 단위를 토큰으로 삼을지"는 사람이 일일이 규칙을 짜기보다는, 데이터를 보고 스스로 배우도록 설계되어 있음.
즉, 토크나이저 학습이란 : 주어진 말뭉치(코퍼스)를 보고, 가장 자주 등장하는 단어나 문자 조합을 사전에 등록하는 과정

예) 코퍼스 : "the cat sat on the mat"
    빈도 높은 단어 : "the", "cat", "sat" -> 통째로 사전에 추가
    드물게 나오는 "mat" 같은 건 "m" + "at" 식으로 subword로 처리
"""
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# 1. 텍스트 파일 불러오기
# Path(".").glob("**/*.txt") : 현재 디렉토리(.)와 하위 디렉토리 전체에서 .txt 파일을 재귀적으로 찾음
# paths : 찾은 모든 .txt 파일의 경로 문자열 리스트
paths = [str(x) for x in Path(".").glob("**/*.txt")]

# 2. 파일 내용 읽기 : 각 파일을 UTF-8 인코딩으로 열고 내용을 읽어서 리스트에 저장.
file_contents = []
for path in paths:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            file_contents.append(file.read())
    except Exception as e:
        print(f"Error reading {path}: {e}")

# 3. 텍스트 하나로 합치기 : 여러 파일에서 읽어온 문자열을 개행 문자(\n)로 이어붙여 하나의 큰 학습 데이터 생성
text = "\n".join(file_contents)

# 토크나이저 초기화
# ByteLevelBPETokenizer : GPT-2 같은 모델에서 사용되는 Byte-Level BPE 방식.
# 장점 : 모든 문자를 바이트 단위로 처리학 때문에 어떤 언어/기호도 손실 없이 다룰 수 있음
tokenizer = ByteLevelBPETokenizer()

# 토크나이저 학습
# train_from iterator : 제공된 텍스트(리스트 형태)에서 BPE 규칙과 어휘집을 학습
# 주요 파라미터 :
"""
    vocab_size = 52_000 : 최종 어휘 크기 설정 (토큰 종류 개수)
    min_frequency = 2 : 최소 2번 이상 등장해야 어휘로 포함됨.
    special_tokens : 모델 학습/추론 시 필요한 특별 토큰 추가
    - <s> : 문장 시작
    - <pad> : 패딩
    - </s> : 문장 끝
    - <unk> : 모르는 단어 (unknown)
    - <mask> : 마스킹 토큰 (예 : MLM 학습용)
"""
tokenizer.train_from_iterator([text], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# 4.2.4 4단계 : 디스크에 파일 저장하기

# 1. 디렉토리 경로 설정
import os
token_dir = './KantaiBERT'

# 2. 디렉토리 생성
if not os.path.exists(token_dir):
  os.makedirs(token_dir)

# 3. 토크나이저 저장
    # - 학습된 ByteLevelBPETokenizer를 실제 파일로 저장
    # - vocab.json : 학습된 어휘(토큰 -> ID 매핑) 저장
    # - merges.txt : BPE 병합 규칙 저장
tokenizer.save_model('KantaiBERT')

# 4.2.5 5단계 : 학습된 토크나이저 파일 로드하기
# ByteLevelBPETokenizer : GPT-2와 같은 방식의 Byte-level BPE 토크나이저 구현체
# BertProcessing : 불러온 토크나이저에 BERT 스타일의 전처리 규칙(예 : [CLS], [SEP]) 을 붙이는 데 사용
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# 학습 후 저장해둔 vocab.json(어휘사전) 과 merges.txt(BPE 규칙)를 불러와 토크나이저를 다시 생성.
# 이 tokenizer 객체는 같은 규칙과 같은 단어 사전을 사용해서 언제든지 동일하게 텍스트 토큰화 가능
tokenizer = ByteLevelBPETokenizer(
    "./KantaiBERT/vocab.json",
    "./KantaiBERT/merges.txt",
)

# 토큰화된 결과 확인 
# encode().tokens() : 입력 문장을 토큰화하고 토큰 문자열 리스트를 반환
print(tokenizer.encode("The Critique of Pure Reason.").tokens)

# 전체 Encoding 객체를 반환
"""
ids : 토큰 ID 리스트
tokens : 토큰 문자열 리스트
offsets : 원문 내 위치 정보
등
"""
print(tokenizer.encode("The Critique of Pure Reason."))


# 후처리기 (Post-Processor) 설정
# BERT 스타일 special token 추가 규칙 설정
# 문장을 토큰화할 때 자동으로:
    # 문장 앞에 <s> 추가
    # 문장 뒤에 </s> 추가
# 인코딩하면 최종 출력은 항상 <s> ... </s> 형태가 됨.
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

# 문장 길이 제한 설정 : 모델 입력 시 길이를 최대 512 토큰으로 제한
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("The Critique of Pure Reason.").tokens)
print(tokenizer.encode("The Critique of Pure Reason."))

# 4.2.6 6단계 : 자원 제약 확인하기

# 4.2.7 7단계 : 모델의 구성 정의하기
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
print(config)

# 4.2.8 8단계 : 트랜스포머의 토크나이저 다시 불러오기
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("./KantaiBERT", max_length =512)

# 4.2.9 9단계 : 모델 초기화
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model)

# 4.2.9.1 파라미터 탐색하기
# 1. 모델의 전체 파라미터 수 출력 : 모델이 학습해야 할 숫자의 총 개수
print(model.num_parameters()) 

# 2. 파라미터 리스트화
"""
- model.parameters() : 모든 파라미터 텐서를 반환 (가중치, bias 등 포함)
- LP : 모든 파라미터를 리스트로 저장
- lp : 파라미터 텐서의 개수 
"""
LP=list(model.parameters())
lp=len(LP)
print(lp)

# 3. 각 파라미터 출력
for p in range(0,lp):
  print(LP[p])
  

# 4. 각 파라미터 텐서의 shape 확인
LP = list(model.parameters())
for i, tensor in enumerate(LP):
    print(f"Shape of tensor {i}: {tensor.shape}")


# 5. 직접 파라미터 수 세기
# 텐서가 2차원(예 : weight matrix)이면 행x열로 파라미터 수 계산
# 텐서가 1차원(예 : bias)이면 그냥 길이(L1)로 계산
# np에 전체 파라미터 수를 누적 
np=0
for p in range(0,lp):#number of tensors
  PL2=True
  try:
    L2=len(LP[p][0]) #check if 2D
  except:
    L2=1             #not 2D but 1D
    PL2=False
  L1=len(LP[p])
  L3=L1*L2
  np+=L3             # number of parameters per tensor
  if PL2==True:
    print(p,L1,L2,L3)  # displaying the sizes of the parameters
  if PL2==False:
    print(p,L1,L3)  # displaying the sizes of the parameters

# 최종 파라미터 수 출력
print(np)          

# 4.2.10 10단계 : 데이터 구축

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./kant.txt",
    block_size=128,
)

# 4.2.11 11단계 : 데이터 콜레이터 정의

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 4.2.12 12단계 : 트레이너 초기화


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=1, #can be increased
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 4.2.13 13단계 : 모델 사전 학습시키기

trainer.train()

# 4.2.14 14단계 : 최종 모델(토크나이저 및 설정 파일) 저장하기

trainer.save_model("./KantaiBERT")

# 4.2.15 15단계 : FillMaskPipeline을 사용한 언어 모델링

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./KantaiBERT",
    tokenizer="./KantaiBERT"
)

fill_mask("Human thinking involves human <mask>.")

# 4.4 정리하기
"""
- KantaiBERT 프로젝트에서 kant.txt의 데이터셋에 대한 토크나이저 학습
- 학습한 merges.txt와 vocab.json 파일 저장
- 사전 학습된 파일로 토크나이저 만듦
- 커스텀 데이터셋을 구축하고 역전파(backpropagation)을 위해 학습 배치를 처리하는 데이터 콜레이터 정의
- 트레이너를 초기화하고 RoBERTa 모델의 파라미터 탐색
- 모델 학습 및 저장
- 저장된 모델을 다운스트림 언어 모델링 작업에 사용
"""