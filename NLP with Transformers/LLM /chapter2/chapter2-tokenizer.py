# 3. 토크나이저
"""
- 토크나이저는 NLP 파이프라인의 핵심 구성 요소 중 하나
- 토크나이저의 목적은 텍스트를 모델이 처리할 수 있는 데이터로 변환하는 것
- 모델은 숫자만 처리할 수 있으므로 토크나이저는 텍스트 입력을 숫자 데이터로 변환해야 함.
- 이를 위한 방법은 여러 가지가 있음.
- 목표는 가장 의미 있는 표현, 즉 모델에 가장 잘 맞는 표현을 찾는 것이고, 가능하다면 가장 작은 표현을 찾는 것
"""
# 3.1 단어 기반
"""
- 일반적으로 몇 가지 규칙만 사용하면 설정과 사용이 매우 쉽고, 종종 괜찮은 결과를 얻을 수 있음
- 텍스트를 분할하는 방법은 여러 가지가 있음
- 예를 들어, Python split() 함수를 적용하여 공백을 사용하여 텍스트를 단어 단위로 토큰화 가능
"""
tokenized_text = "짐 헨슨은 인형극 배우였습니다".split()
print(tokenized_text)

# 3.2 문자 기반
"""
- 텍스트를 단어가 아닌 문자로 분할
    주요 이점
    - 어휘가 훨씬 적음
    - 모든 단어는 문자로 구성될 수 있으므로 어휘에서 벗어난(알 수 없는) 토큰은 훨씬 적음

    주요 의문
    - 문자를 기반으로 표현되기 때문에 직관적으로 의미가 덜함.
    - 모델이 처리해야 할 토큰의 양이 매우 많아짐.
        단어는 단어 기반 토크나이저를 사용하면 단일 토큰에 불과하지만, 문자로 변환하면 쉽게 10개 이상의 토큰으로 바뀔 수 있음.

- 단어 기반과 문자 기반의 장점을 모두 얻으려면 두 가지 접근 방식을 결합한 세 번째 기술인 하위 단어 토큰화를 사용 가능
"""

# 3.3 하위 단어 토큰화
"""
- 자주 사용되는 단어를 더 작은 하위 단어로 나누지 않고, 드물게 사용되는 단어를 의미 있는 하위 단어로 분해해야 한다는 원칙에 의존
- 예) "annoyingly"는 희귀한 단어로 간주되어 "annoying"과 "ly"로 분해될 수 있음
- 이 두 단어는 독립적인 부사로 더 자주 등장할 가능성이 높지만, 동시에 "annoyingly"의 의미는 "annoying"과 "ly"의 합성어로 유지됨
"""

"""
- Let's</w> do</w> token ization</w> !</w>

- 위의 예에서 "tokenization"은 "token"과 "ization" 으로 나우었음
- 이 두 토큰은 의미를 가지면서도 공간 효율적(긴 단어를 표현하는데 토큰이 두 개만 필요함.)
- 이를 통해 적은 어휘로도 비교적 좋은 커버리지를 확보할 수 있으며, 알려지지 않은 토큰은 거의 없음.

- 예시)
    - GPT-2에서 사용되는 바이트 수준 BPE
    - BERT에서 사용되는 WordPiece
    - 여러 다국어 모델에서 사용되는 SentencePiece 또는 Unigram

"""
# 3.4 로딩 및 저장

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-case")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenizer("Using a Transformer network is simple")

tokenizer.save_pretrained("directory_on_my_computer")

# 3.5 encoding
"""
- 인코딩 : 텍스트를 숫자로 변환하는 것
- 토큰화와 입력 ID로의 변환이라는 두 단계로 진행됨
- 첫 번째 단계는 텍스트를 단어(또는 단어의 일부, 구두점 기호 등)로 분할하는 것. 이를 토큰이라고 함.
- 이 과정에는 여러 규칙이 적용되므로, 모델 이름을 사용하여 토크나이저를 인스턴스화해야 함.
- 이렇게 하면 모델이 사전 학습될 때 사용된 규칙과 동일한 규칙을 사용 가능
- 두 번째 단계는 토큰을 숫자로 변환하여 텐서를 생성하고 모델에 전달하는 것
- 이를 위해 토크나이저는 어휘(vocabulary)를 가지고 있는데, 이는 메서드로 토크나이저를 인스턴스화할 때 다운로드하는 부분.
"""
# 3.5.1 Tokenization

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(sequence)

print(tokens)

#['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

# 3.5.2 From tokens to input IDs
"""
- 입력 ID 로의 변환은 tokenizer의 convert_tokens_to_ids() 메서드에서 처리됨
"""

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
# [7993, 170, 11303, 1200, 2443, 1110, 3014]
# 텐서로 변환된 후 모델의 입력으로 사용 가능

# 3.5.3 Decoding
"""
- 디코딩은 반대 방향으로 진행됨. 어휘 인덱스에서 문자열을 얻고자 함.
- 이는 다음과 같이 decode() 메서드로 수행 가능
"""

decoded_string = tokenizer.decode([7991,170,11303,1200,2443,1110,3014])
print(dec)

# 'Using a Transformer network is simple'

"""
- 이 decode 방법은 색인을 토큰으로 다시 변환할 뿐 아니라, 같은 단어에 속했던 토큰들을 그룹화하여 읽을 수 있는 문장을 생성
- 이러한 동작은 새로운 텍스트(프롬프트에서 생성된 텍스트 또는 번역이나 요약과 같은 시퀀스 간 문제)를 예측하는 모델을 사용할 때 매우 유용
"""