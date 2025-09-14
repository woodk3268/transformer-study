# 토큰화기를 처음부터 구축하기 : 데이터셋으로 새로운 토큰화기를 훈련
"""
- 훈련 코퍼스에 존재하는 모든 문자를 토큰으로 시작
- 가장 흔한 토큰 쌍을 식별하여 하나의 토큰으로 병합
- 어휘 사전이 원하는 크기에 도달할 때까지 반복
- 라이브러리
"""
from datasets import load_dataset

# “wikitext-103-raw-v1” 옵션 또는 “Salesforce/wikitext”, “wikitext-103-v1”
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_texts = dataset["train"]["text"]

from tokenizers import Tokenizer
from tokenizers.models import BPE
# 라이브러리의 주요 API : Tokenizer 클래스. BPE 모델로 인스턴스화하는 방법
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Trainer = 학습 설정(어휘 크기, 특수 토큰, 최소 빈도 등)을 지정하는 훈련 도우미
from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 입력을 단어로 분할하는 사전 토큰화기 없이 훈련하면 여러 단어가 겹치는 토큰이 생성될 수 있음.
# 예를 들어 "it is" 토큰이 생성될 수 있는데, 이 두 단어는 종종 인접하여 나타남.
# 사전 토큰화기를 사용하면 어떤 토큰도 사전 토큰화기가 반환하는 단어보다 크지 않게 보장됨.
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer= Whitespace()

files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)

tokenizer.save("data/tokenizer-wiki.json")

tokenizer = Tokenizer.from_file("/data/tokenizer-wiki.json")

#  텍스트에 토큰화기의 전체 파이프라인을 적용하여 Encoding 객체를 반환
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")

# tokens : 텍스트를 토큰으로 분할한 결과
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]


# ids : 각 토큰의 인덱스
print(output.ids)
# [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]


# 토큰화 라이브러리의 중요한 특징 : 완전한 정렬 추적 기능 제공
# 주어진 토큰에 해당하는 원본 문장의 부분을 항상 확인 가능
# Encoding 객체의 offset 속성에 저장
# 예를 들어, 목록에서 인덱스 9에 위치한 토큰인 "[UNK]"가 발생한 원인을 찾고자 한다면, 해당 인덱스의 오프셋을 요청하면 됨.

print(output.offsets[9])
# (26, 27)

sentence = "Hello, y'all! How are you 😁 ?"
sentence[26:27]

# post-processing
# 토큰화 도구가 "[CLS]"나 "[SEP]" 같은 특수 토큰을 자동으로 추가하도록 설정 가능
# 이를 위해 후처리기를 사용
# TemplateProcessing이 주로 사용되며, 단일 문장 및 문장 쌍 처리용 템플릿과 특수 토큰 및 해당 ID 지정해서 사용
# 토큰화기를 구축할 때 특수 토큰 목록의 1번과 2번 위치에 "[CLS]"와 "[SEP]"를 설정했으므로, 이 값들이 해당 토큰의 ID여야 함. (Tokenizer.token_to_id method로 확인가능)
tokenizer.token_to_id("[SEP]")

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]



from tokenizer.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single = "[CLS] $A [SEP]",
    pair = "[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens = [
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]

output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
print(output.tokens)
# ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]

# 토큰에 할당된 유형 ID가 올바른지 확인
print(output.type_ids)
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Encoding multiple sentences in a batch
output = tokenizer.encode_batch(
    [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
)

# 여러 문장을 인코딩할 때 Tokenizer.enable_padding을 사용하면 출력을 가장 긴 문장의 길이로 자동 패딩 
# 이때 pad_token과 해당 ID를 지정해야 함.
tokenizer.enable_padding(pad_id = 3, pad_token = "[PAD]")
output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
print(output[1].tokens)
# ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]

print(output[1].attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 0]


# pretrained
# 사전 훈련된 토큰화기 사용
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase = True)
