
# The tokenization pipeline
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")

# normalization
# 공백 제거, 전체 텍스트 소문자 변환 등
# 각 정규화 작업은 Tokenizers 라이브러리에서 Normalizer로 표현되며, normalizers.Sequence를 사용해 여러 작업 결합 가능

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
normalizer = normalizers.Sequence([NFD(), StripAccents()])

normalizer.normalize_str("Héllò hôw are ü?")
# "Hello how are u?"

tokenizer.normalizer = normalizer


# Pre-Tokenization
# 텍스트를 더 작은 객체로 분할하여 훈련 종료 시 최종 토큰의 상한선을 설정하는 과정
# 사전 토큰화기가 텍스트를 "단어"로 분할한 후, 최종 토큰이 해당 단어의 일부가 됨
# 입력을 사전 토큰화하는 쉬운 방법 : 공백과 구두점을 기준으로 분할

from tokenizers.pre_tokenizers import Whitespace
pre_tokenizer = Whitespace()
pre_tokenizer.pre_tokenize_str("Hello! How are you? I'm fine, thank you.")
# [("Hello", (0, 5)), ("!", (5, 6)), ("How", (7, 10)), ("are", (11, 14)), ("you", (15, 18)),
#  ("?", (18, 19)), ("I", (20, 21)), ("'", (21, 22)), ('m', (22, 23)), ("fine", (24, 28)),
#  (",", (28, 29)), ("thank", (30, 35)), ("you", (36, 39)), (".", (39, 40))]

# 출력은 튜플 목록으로, 각 튜플은 하나의 단어와 원본 문장 내 해당 단어의 span을 포함함.

from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
pre_tokenizer.pre_tokenize_str("Call 911!")
# [("Call", (0, 4)), ("9", (5, 6)), ("1", (6, 7)), ("1", (7, 8)), ("!", (8, 9))]

tokenizer.pre_tokenizer = pre_tokenizer

# model
# 입력 텍스트가 정규화되고 사전 토큰화되면 토크나이저는 사전 토큰에 모델 적용
# 이 파이프라인 단계는 사용자의 코퍼스로 훈련해야하는 부분 (또는 사전 훈련된 토크나이저를 사용하는 경우 이미 훈련된 상태)
# 모델의 역할은 학습한 규칙을 사용하여 "단어"를 토큰으로 분할
# 또한 해당 토큰들을 모델 어휘사전 내 대응하는 ID로 매핑하는 역할도 담당
"""
models.BPE
models.Unigram
models.WordLevel
models.WordPiece
"""

# post-processing
"""
후처리(post-processing)는 토큰화 파이프라인의 마지막 단계로, 인코딩이 변환 되기 전에 잠재적 특수 토큰 추가와 같은 추가 변환 수행
"""
from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

# 토큰화 전처리기나 정규화기와 달리 후처리기를 변경한 후, 토큰화기를 재훈련할 필요가 없음.

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
bert_tokenizer = Tokenizer(WordPiece(unk_token = "[UNK]"))

from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])


from tokenizers.pre_tokenizers import Whitespace
bert_tokenizer.pre_tokenizer = Whitespace()

from tokenizers.processors import TemplateProcessing
bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
bert_tokenizer.train(files, trainer)
bert_tokenizer.save("data/bert-wiki.json")

# 디코딩
# 입력 텍스트를 인코딩하는 것 외에도, 토큰화기는 디코딩, 즉 모델이 생성한 ID를 다시 텍스트로 변환하는 API 제공
# 디코더는 먼저 ID를 토큰으로 변환(토크나이저의 어휘 사전 사용)하고 모든 특수 토큰을 제거한 후, 해당 토큰들을 공백으로 연결

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.ids)
# [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]
tokenizer.decode([1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2])
# "Hello , y ' all ! How are you ?"

output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.")
print(output.tokens)
# ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]
bert_tokenizer.decode(output.ids)
# "welcome to the tok ##eni ##zer ##s library ."

from tokenizers import decoders
bert_tokenizer.decoder = decoders.WordPiece()
bert_tokenizer.decode(output.ids)
# "welcome to the tokenizers library."