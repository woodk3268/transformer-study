# 트랜스포머로 시작하는 자연어 처리
# 2장 트랜스포머 모델 아키텍처 살펴보기

# 2.1 트랜스포머의 시작 : Attention is All You Need
"""
1. 트랜스포머 구조
- 오리지널 트랜스포머는 층 여섯 개를 쌓아 올린 스택 형태
- 마지막 층을 제외하고 N번째 층의 출력은 N+1번째 층의 입력이 됨.
- 왼쪽에는 여섯 개의 층을 가진 인코더 스택이 있고, 오른쪽에는 여섯 개의 층을 가진 디코더 스택이 있음
- 왼쪽은 트랜스포머의 인코더 부분으로 입력값이 들어오는 부분이다. 어텐션 층과 순방향(feedforward) 층으로 이루어져 있음.
- 오른쪽은 두 어텐션 층과 하나의 순방향 층으로 이루어진 트랜스포머의 디코더 부분으로, 타깃(target) 출력값을 입력받음.
- RNN, LSTM, CNN 등은 전혀 사용하지 않았음.
- 트랜스포머에는 재귀적(recurrence) 구조가 없음.

2. 어텐션
- 단어 간 거리가 멀어질수록 더 많은 파라미터가 필요했던 재귀적 구조 대신 어텐션을 사용했음.
- 어텐션은 "단어-투-단어(word to word)" 연산 (실제로는 토큰-투-토큰(token to token) 연산)
- 어텐션 매커니즘은 한 단어가 자신을 포함한 시퀀스 내 모든 단어들과 각각 어떻게 연관되어 있는지 계산.

예)
'The cat sat on the mat.'

- 어텐션은 단어 벡터 간의 내적(dot product)을 사용하여 한 단어와 가장 밀접한 관계를 가지는 단어를 찾음.
- 이때 탐색 대상에는 자기 자신도 포함됨("cat"과 "cat" 사이의 관계)

3. 멀티-헤드 어텐션
- 시퀀스에 대한 더 심층적인 분석
- 재귀적 구조를 없애 계산량 감소
- 병렬화로 인한 학습 시간 단축
- 동일한 입력 시퀀스를 다른 관점으로 학습하는 각각의 어텐션 메커니즘
"""

# 2.1.1 인코더 스택
"""
1. 인코더 구조
- 오리지널 트랜스포머 모델의 인코더 층은 총 6개이고 모두 동일한 구조
- 각각의 층에 멀티-헤드 어텐션 메커니즘, 완전 연결 위치별 순방향 네트워크(fully connected position-wise feed-forward network)인 두 서브 층을 가지고 있음.
- 잔차 연결(residual connection)이 트랜스포머 모델의 각 서브 층을 둘러싸고 있음.
- 잔차 연결은 서브 층의 입력 x를 층 정규화(layer normalization) 함수에 전달하여, 위치 인코딩(positional encoding)과 같은 중요한 정보가 손실되지 않도록 보장.
- 각 층의 정규화된 출력은 다음과 같다.

LayerNormalization(x + Sublayer(x))

2. 각 층의 역할
- 인코더의 N=6개 층이 모두 완전히 동일한 구조일지라도, 각 층은 서로 다른 내용을 담고 있다.
- 예를 들어, 임베딩 서브 층은 스택의 가장 아래에만 위치한다. 다른 다섯 층은 임베딩 층을 포함하고 있지 않고, 
    이 덕분에 인코딩된 입력이 모든 층에 걸쳐 안정적으로 유지된다.

- 멀티-헤드 어텐션 메커니즘 또한 여섯 개의 층에 동일하게 적용되지만 각자 다른 역할을 수행한다.
- 각 층은 이전 층의 출력을 토대로 시퀀스 내 토큰들의 관계를 파악할 다양한 방법들을 학습한다.

3. 제약 조건
- 임베딩 층과 잔차 연결을 포함해서 모델을 구성하는 모든 서브 층의 출력 차원을 일정하게 함.
- 모델을 구성하는 모든 서브 층의 출력 차원을 일정하게 함
- 이 차원(d_model)은 목적에 따라 다른 값으로 설정할 수 있으며 오리지널 트랜스포머 모델에서는 d_model = 512로 설정
- 출력 차원 d_model을 항상 일정하게 유지할 수 있게 됨에 따라, 연산량과 리소스의 사용량을 줄이고 모델에 흐르는 정보를 쉽게 추적 가능

"""

# 2.1.1.1 입력 임베딩
"""
1. 임베딩 서브 층
- 입력 임베딩 서브 층은 오리지널 트랜스포머 모델의 학습된 임베딩을 사용하여 입력 토큰을 d_model = 512 차원의 벡터로 변환.
- 임베딩 서브 층은 일반적인 트랜스덕션(transduction) 모델과 동일하게 동작
- 먼저 BPE(Byte-Pair Encoding) 워드 피스(word piece), 센텐스 피스(sentence piece)와 같은 토크나이저가 문장을 토큰으로 분리.


예)
"the Transformer is an innovative NLP model!"
-> ['the', 'transform', 'er', 'is', 'an', 'innovative', 'n','l','p', 'model', '!']

토크나이저가 대문자를 소문자로 변경하고 문장을 하위 부분들로 잘라냄.
일반적으로 토크나이저는 다음과 같이 임베딩 과정에 사용될 정수 표현까지 제공.

text = "The cat slept on the couch. It was too tired to get up."
tokenized text = [1996, 4937, ..., 1012]

2. 임베딩
스킵 그램(skip-gram) : 주어진 단어에 기초하여 주변(context) 단어를 예측하도록 학습하는 모델
스텝 크기가 2인 윈도우(window)의 중심에 단어 word(i)가 있다면 word(i-2), word(i-1), word(i+1), word(i+2)를 학습하고, 
윈도우를 한 칸씩 움직이며 과정 반복.
스킵 그램 모델은 일반적으로 입력 층, 가중치, 은닉 층, 토큰화된 입력단어에 대한 임베딩 출력 층으로 구성됨.

예)
"The black cat sat on the couch and the brown dog slept on the rug."

'black' 과 'brown' 두 단어를 살펴보면, 두 단어의 임베딩 벡터는 비슷할 것.
우리는 각 단어에 대한 d_model = 512 차원의 벡터를 생성해야 하므로, 각 단어마다 512 차원의 임베딩 벡터를 얻을 것.

코사인 유사도(cosine similarity)로 'black'과 'brown'의 임베딩이 유사한지 확인하면, 두 단어의 임베딩 결과 검증 가능

-> 단어들이 어떻게 연관되는지 단어 임베딩으로 학습.

3. 위치 인코딩

- 위치 벡터를 별개로 학습하면 트랜스포머의 학습 속도가 매우 느려질 수 있고 어텐션의 서브 층이 너무 복잡해질 위험이 있음.
- 따라서 추가적인 벡터를 사용하는 대신, 입력 임베딩에 위치 인코딩 값을 더하여 시퀀스 내 토큰의 위치 표현
- 위치 임베딩 함수의 출력 벡터는 d_model = 512(또는 설정한 다른 상수)의 고정된 크기로 트랜스포머에 전달되어야 함.
- 단어 임베딩 서브 층에 사용한 문장을 돌아보면, 'black', 'brown'이 의미적으로 비슷하지만, 문장 내에선 멀리 떨어져 있음.
- 'black' 이라는 단어는 두 번째(pos = 2)에 위치에 있으며 'brown'은 열 번째(pos=10)에 위치함.
- 우리는 각각의 단어 임베딩에 적절한 값을 더해 정보를 추가해야 함.
- 정보가 더해져야 할 벡터의 크기는 d_model = 512 차원이므로, 512 개의 숫자를 사용해 'black'과 'brown'의 단어 임베딩 벡터에 위치 정보를 주어야 함.
- 위치 임베딩을 구현하는 방법에는 여러 가지가 있음.
- 사인과 코사인으로 각 위치와, 단어 임베딩의 각 차원 d_model = 512개에 서로 다른 주기를 가지는 위치 인코딩(PE) 값을 생성함.
- 단어 임베딩의 맨 앞 차원부터 시작해서, i=0 부터 i=511 까지 순서대로 적용함.
- 이때, 짝수 번째는 사인 함수를, 홀수 번째는 코사인 함수를 적용

"""

# import math

# def positional_encoding(pos, pe):
#     for i in range(0, 512,2):
#         pe[0][i] = math.sin(pos/(10000** ((2*i)/d_model)))
#         pe[0][i+1] = math.cos(pos/(10000 ** ((2*i)/d_model)))
#     return pe

"""
4. 임베딩 벡터에 위치 인코딩 더하기

예)
- 'black'의 단어 임베딩 y1 = black을, 인코딩 함수로 얻은 위치 벡터 pe(2)와 더한다고 하자.
- 입력 단어 'black'에 대한 위치 인코딩 pc(black)은 다음과 같이 얻는다.
pc(black) = y1 + pe(2)

- 이렇게 더하기만 한다면, 위치 정보로 인해 단어 임베딩의 정보가 훼손될 위험이 있음.
- 단어 임베딩 층의 정보를 이어지는 층에 더 확실하게 전달하기 위해, y1의 값을 키우는 다양한 방법 존재
- 그중 한 가지는, 'black'의 임베딩 y1에 임의의 값을 곱하는 것

y1 * math.sqrt(d_model)

- 이제 동일한 512 크기의 두 벡터, 'black'의 단어 임베딩과 위치 벡터를 더함
"""

# for i in range(0,512, 2):
#     pe[0][i] = math.sin(pos/ (10000 ** ((2*i)/d_model)))
#     pc[0][i] = (y[0][i]*math.sqrt(d_model)) + pe[0][i]

#     pe[0][i+1] = math.cos(pos / (10000 ** ((2*i)/d_model)))
#     pc[0][i+1] = (y[0][i+1]*math.sqrt(d_model))+ pe[0][i+1]


# 2.1.1.2 서브층 1 : 멀티-헤드 어텐션
"""
1. 멀티-헤드 어텐션 서브층
- 여덟 개의 헤드가 있으며 포스트-층 정규화(post-layer normalization)와 이어져있음.
- 포스트- 층 정규화는 서브 층의 출력을 잔차 연결과 더한 후 정규화함.
"""

# 2.1.1.3 멀티-헤드 어텐션 아키텍처
"""
- 인코더 스택 첫 번째 층의 멀티 어텐션 서브 층으로 각 단어의 임베딩과 위치가 담겨있는 벡터가 입력됨
- 인코더 스택의 이어지는 층에는 이 정보들이 다시 입력되지 않음
- 입력 시퀀스의 각 단어 x_n 을 표현하는 벡터의 크기는 d_model = 512
- 이제 각 단어 x_n을 d_model=512 차원의 벡터로 표현 가능
- 각각의 단어를 다른 모든 단어와 비교하여 시퀀스에 얼마나 적합할 지 결정

예)
다음 문장에서, 'it'은 'cat'일 수도 또는 'rug' 일 수도 있다.

Sequence = The cat sat on the rug and it was dry-cleaned.

- 모델은 학습과정에서 'it'이 'cat'과 연관되어 있는지 아니면 'rug'와 연관되어 있는지 찾으려 할 것
- 설정해 둔 d_model = 512 차원의 모델로 대규모 학습을 수행하면 됨.
- 그런데 하나의 d_model 차원 블록만 사용하면 한 번에 하나의 관점으로만 시퀀스를 분석하게 됨.
- 대신, 시퀀스 내 단어 집합 x의 각 단어 x_n을 표현하는 d_model = 512 차원을 여덟 개로 나누어 d_k = 64 차원으로 만들면 더 효과적
- "헤드" 여덟 개를 병렬로 연산하면 학습 속도를 높이면서 단어 간의 관계를 표현하는 서로 다른 여덟 개의 표현 공간(representation subspace)를 얻게 됨.

- 각 헤드의 출력을 x * d_k 모양의 행렬 Z_i 라고 한다면, 멀티-헤드 어텐션의 출력 Z는 다음과 같다.
Z = (Z_0, Z_1, Z_2, Z_3, Z_4, Z_5, Z_6, Z_7)

- 멀티-헤드 서브 층의 출력이 벡터 시퀀스가 아니라 x_m * d_model 형태의 행렬이 될 수 있도록 연결(concatenate)해야 함.
- 멀티-헤드 어텐션 서브 층을 벗어나기 전에, Z의 요소들을 연결하면 다음과 같다.

MultiHead(output) = Concat(Z_0, Z_1, Z_2, Z_3, Z_4, Z_5, Z_6, Z_7) = x, d_model

- 각각의 헤드를 서로 연결하여 d_model = 512 차원이 되었다. 

- 어텐션 메커니즘 헤드 h_h의 내부에서는 "단어"를 세 가지 행렬로 표현한다.
    - 다른 "단어" 행렬의 모든 키-밸류 쌍(key-value pair)을 탐색하는 d_q =64 차원의 쿼리(query) 행렬(Q)
    - 어텐션 점수를 구하기 위해 학습한, d_k = 64차원의 키(key) 행렬 (K)
    - 또 다른 어텐션 점수를 구하기 위해 학습한 d_v = 64 차원의 밸류(value) 행렬 (V)

- 스케일드 내적 어텐션 메커니즘은 다음과 같이 Q, K, V 로 표현 가능
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

- Q, K, V 는 동일한 차원을 가지므로, 내적 연산으로 여덟 개의 헤드에서 어텐션 값을 계산하고 서로 연결하여 출력 Z를 얻는 과정이 비교적 간단.
- Q, K, V 를 얻기 위해서는 d_model = 512 개의 행과 d_k = 64 개의 열을 가진 가중치 행렬 Q_w, K_w, V_w 를 학습해야 함
- Q는 x와 Q_w의 내적으로 얻을 수 있고, 결과적으로 Q의 차원은 d_k = 64가 됨

"""

# 2.1.1.4 1단계 : 입력 표현
import numpy as np
from scipy.special import softmax


print("Step 1 : Input : 3 inputs, d_model = 4")
x = np.array(
    [[1.0, 0.0, 1.0, 0.0], # 입력1
    [0.0, 2.0, 0.0, 2.0], # 입력2
    [1.0, 1.0, 1.0, 1.0] # 입력3
    ]
)
print(x)

# 2.1.1.5 2단계 : 가중치 행렬 초기화
"""
    - 각 입력에는 세 개의 가중치 행렬이 관여
        - 쿼리를 얻기 위한 Q_w
        - 키를 얻기 위한 K_w
        - 밸류를 얻기 위한 V_w

    - 세 가지 가중치 행렬은 모델의 모든 입력에 적용된다.
"""

print("Step 2 :weights 3 dimensions x d_model = 4")
print("w_query")
w_query = np.array(
    [[1,0,1],
    [1,0,0],
    [0,0,1],
    [0,1,1]]
)
print(w_query)

print("w_key")
w_key = np.array(
    [[0,0,1],
    [1,1,0],
    [0,1,0],
    [1,1,0]]
)
print(w_key)

print("w_value")
w_value = np.array(
    [[0,2,0],
    [0,3,0],
    [1,0,3],
    [1,1,0]]
)
print(w_value)

# 2.1.1.6 3단계 : Q, K, V를 얻기 위한 행렬 곱
"""
    - 입력 벡터에 가중치 행렬을 곱해 각 입력에 대한 쿼리, 키, 밸류 백털르 얻는다.
    - 이 모델에서는 모든 입력에 대해 w_query, w_key, w_value 가중치 행렬을 하나씩만 사용한다고 가정한다.
"""

print("Step 3 : Matrix multiplication to obtain Q, K, V")
print("Query: x * w_query")
Q = np.matmul(x, w_query)
print(Q)


print("Key: x * w_key")
K = np.matmul(x, w_key)
print(K)


print("Value: x * w_value")
V = np.matmul(x, w_value)
print(V)

# 2.1.1.7 4단계 : 스케일드 어텐션 점수
print("Step 4 : Scaled Attention Scores")
k_d = 1 # k_d = 3 의 제곱근을 내림하여 1을 사용
attention_scores = (Q @ K.transpose())/ k_d
print(attention_scores)

# 2.1.1.8 5단계 : 각 벡터의 스케일드 소프트맥스 어텐션 점수
print("Step 5 : Scaled softmax attention_scores for each vector")
attention_scores[0] = softmax(attention_scores[0])
attention_scores[1] = softmax(attention_scores[1])
attention_scores[2] = softmax(attention_scores[2])

print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])

# 2.1.1.9 6단계 : 어텐션을 표현하기 위한 마지막 과정
print("Step 6 : attention value obtained by score1/k_d*V")
print(V[0])
print(V[1])
print(V[2])

print(attention_scores[0].shape)
attention1 = attention_scores[0].reshape(-1, 1)
print(attention1.shape)

print("Attention 1")
attention1 = attention_scores[0][0]*V[0]
print(attention1)

print("Attention 2")
attention2 = attention_scores[0][1]*V[1]
print(attention2)

print("Attention 3")
attention3 = attention_scores[0][2]*V[2]
print(attention3)

# 2.1.1.10 7단계 : 결과 합산하기
print("Step 7 : summed the results to create the first line of the output matrix")
attention_input1 = attention1 + attention2 + attention3
print(attention_input1)

# 2.1.1.11 8단계 : 1단계부터 7단계까지의 과정을 모든 입력에 적용

print("Step 8 : Step 1 to 7 for inputs 1 to 3")
# 학습된 가중치로 결과 세 개를 얻었다고 가정한다.
# 오리지널 트랜스포머 논문대로 구현했다고 가정한다. 64 차원의 결과 세 개를 만든다.
attention_head1 = np.random.random((3,64))
print(attention_head1)

# 2.1.1.12 9단계 : 어텐션 서브 층 헤드의 출력
# 어텐션 서브 층의 여덟 헤드를 이미 학습했다고 가정
# 각 트랜스포머 헤드는(세 개의 단어 또는 워드피스에 대해) d_model= 64 차원의 벡터 세 개를 출력
print("Step 9 : We assume we have trained the 8 heads of the attention sublayer")
z0h1 = np.random.random((3,64))
z1h2 = np.random.random((3,64))
z2h3 = np.random.random((3,64))
z3h4 = np.random.random((3,64))
z4h5 = np.random.random((3,64))
z5h6 = np.random.random((3,64))
z6h7 = np.random.random((3,64))
z7h8 = np.random.random((3,64))

print("shape of one head", z0h1.shape, "dimension of 8 heads", 64*8)

# Z = (Z_0, Z_1, Z_2, Z_3, Z_4, Z_5, Z_6, Z_7)

# 2.1.1.13 10 단계 : 각 헤드의 출력 연결하기
# MultiHead(Output) = Concat(Z_0, Z_1, Z_2, Z_3, Z_4, Z_5, Z_6, Z_7)W^0 = x, d_model
# Z를 또 다른 학습된 가중치 행렬 W^0 와 곱함

print("Step 10 : Connection of heads 1 to 8 to obtain the original 8X64 = 512 output dimension of the model")
output_attention = np.hstack((z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8))
print(output_attention)

# 2.1.1.14 포스트-층 정규화
"""
- 포스트-층 정규화는 덧셈 함수와 층 정규화 작업으로 구성
- 덧셈 함수는 서브 층의 입력과 이어진 잔차 연결을 처리
- 잔차 연결은 중요한 정보들을 잃어버리지 않도록 방지

LayerNormalization(x + Sublayer(x))

- Sublayer(x)는 서브 층을 의미. x는 서브 층에 입력된 정보
- x + Sublayer(x)의 결과인 벡터 v가 LayerNormalization의 입력이 됨
- 트랜스포머의 모든 입력과 출력은 모든 과정에서 d_model = 512 차원으로 표준화(standardize)됨.

- LayerNormalization(v) = gamma * (v-mu)/sigma + beta

- mu : d차원 벡터 v에 대한 평균
- sigma : d차원 벡터 v에 대한 분산 
- gamma : 스케일링을 위한 파라미터
- beta : 편향(bias) 벡터
"""

# 2.1.1.15 서브 층 2 : 순방향 네트워크
"""
순방향 네트워크 (FFN)의 입력은 d_model = 512 차원으로, 앞선 서브 층에 대한 포스트-층 정규화의 출력

- 인코더와 디코더의 순방향 네트워크는 완전 연결(fully connected) 되어있다.
- 순방향 네트워크는 위치별(position-wise) 네트워크. 동일한 연산을 위치별로 각각 수행
- 순방향 네트워크는 층 두 개와 ReLU 할성화 함수로 구성
- 순방향 네트워크 층의 입력과 출력은 d_model = 512 차원이지만, 내부 층은 d_ff = 2048로 더 크다.
- 순방향 네트워크는 크기가 1인 커널로 두 번의 컨볼루션 연산을 수행하는 것으로 볼 수 있다.


FFN(x) = max(0, xW_1 + b_1) W_2 + b_2

- 순방향 네트워크의 출력은 포스트-층 정규화로 이어짐
- 그러고 나서, 결과 값은 인코더 스택의 다음 층과 디코더 스택의 멀티-헤드 어텐션 층으로 전달됨
"""

# 2.1.2 디코더 스택
"""
- 트랜스포머의 디코더 역시 인코더처럼 층을 쌓아 올린 스택 형태

1. 디코더 각 층의 구조
- N = 6 개 디코더 층의 구조는 모두 동일
- 각 층은 3개의 서브 층으로 이루어져 있는데, 멀티-헤드 마스크드 어텐션(multi-head masked attention) 메커니즘, 멀티-헤드 어텐션 메커니즘, 완전 연결 위치별 순방향 네트워크로 구성
- masked multi head attention : 주어진 위치 이후의 모든 단어를 마스킹함으로써, 트랜스포머가 나머지 시퀀스를 보지 않고 스스로의 추론에 근거하여 연산
- 인코더 스택처럼, 세 개의 주요 서브 층 각각을 잔차 연결과 Sublayer(x) 가 감싸고 있음.

LayerNormalization(x + Sublayer(x))

- 인코더 스택에서처럼, 임베딩 서브 층은 디코더 스택의 가장 아래층과 연결되어 있으며 임베딩 층과 잔차 연결을 포함한 모든 서브 층의 출력은 d_model 차원으로 일정
"""

# 2.1.2.1 출력 임베딩과 위치 인코딩
"""
출력 임베딩 층과 위치 인코딩 함수는 인코더 스택에서와 동일
"""

# 2.1.2.2 어텐션 층
"""
 1. 인코더 vs 디코더 어텐션

* 인코더: 모든 입력 토큰이 서로를 자유롭게 바라봄 (self-attention).
* 디코더: 출력을 하나씩 예측해야 하므로 앞에 나온 토큰만 참고할 수 있음 → 이것이 마스크드(masked) self-attention.

즉:

* 인코더 self-attention: 전체 시퀀스에서 자유롭게
* 디코더 self-attention: 현재 위치 이전까지만 (뒤는 가려짐)


2. 디코더 안에는 어텐션이 두 번

    1. Masked Multi-Head Self-Attention

    * 디코더 입력 토큰들끼리만 attention.
    * 마스크 때문에 뒤쪽 정보는 못 봄 (auto-regressive 성질 보장).

    Input_Attention = (Output_decoder_sub_layer-1 (Q), Output_encoder_layer (K, V))

    - Output_decoder_sub_layer-1 (Q) : 디코더의 바로 전 단계 출력으로 만든 Query (즉, 현재까지 생성된 토큰들이 "무엇을 찾고 싶은지" 신호를 보냄)
    - Output_encoder_layer (K, V) : 인코더의 최종 출력으로 만든 Key, Valu (즉, 입력 문장 전체가 가진 정보)

    2. Encoder-Decoder Attention

    * Query = 디코더 (현재 토큰)
    * Key, Value = 인코더 출력
    * 즉, 디코더가 인코더에서 추출된 문맥 정보를 참고하는 단계.

---

3. 정규화와 연결

* 각 서브 층 뒤에는 residual connection + layer normalization이 붙음.
* 그래서 흐름은:

  ```
  디코더 입력 
   → (마스크드 self-attention) 
   → (residual + norm) 
   → (encoder-decoder attention) 
   → (residual + norm) 
   → (feed-forward network) 
   → (residual + norm)
  ```

"""

# 2.1.2.3 순방향 서브 층, 포스트-층 정규화 그리고 선형 층
"""
- 트랜스포머는 한 번에 하나의 출력 시퀀스만 생성
    Output sequence = (y_1, y_2, ..., y_n)

- 선형 층의 출력을 생성하는 선형 함수(linear function)은 모델마다 다르지만, 표준 식은 다음과 같다.
    y = w * x + b (w, b : 학습 파라미터)

- 선형 층은 시퀀스의 다음으로 등장할 법한 요소들을 예측하고, 소프트맥스 함수로 가장 가능성 있는 요소가 정해짐
- 인코더 층과 마찬가지로 디코더 층 또한 l번째 층에서 l+1 번째 층으로 N=6 개 중 최상위 층까지 이어짐
"""

# 2.2 학습과 성능
"""
- 아담 옵티마이저를 사용했지만, 워밍업 단계에서 학습률을 일정하게 증가시키고 이후 조금씩 감소시키는 방식
- 임베딩을 합치는 과정에는 드롭아웃(dropout) 및 잔차 드롭아웃(residual dropout) 같은 다양한 정규화 기법 적용
- 또한, 원-핫(one-hot)에 대한 과신(overconfident)과 오버피팅(overfitting)을 방지하기 위해 라벨 스무딩(label smoothing)이 적용됨
- 라벨 스무딩을 사용하면 평가 과정의 정확도가 낮아지지만, 모델이 더 많이 더 잘 학습하게 됨.
"""

# 2.3 허깅페이스의 트랜스포머 모델

#@title pipeline 모듈을 불러오고 영어 프랑스어 번역을 선택한다.
from transformers import pipeline
translator = pipeline("translation_en_to_fr") # 한 줄의 코드면 충분
print(translator("It is easy to translate languages with transformers", max_length=40))

# 2.4 정리하기
"""
- RNN, LSTM, CNN 등이 없이도 트랜스덕션과 시퀀스 모델링이 가능하게 한 트랜스포머 아키텍처의 접근 방식
- 인코더와 디코더의 크기를 표준화하고 대칭적인 설계를 한 덕분에 여러 서브 층을 매끄럽게 연결
- 트랜스포머가 순환 신경망을 제거하는 것 외에도 병렬화를 적극 도입하여 학습 시간을 단축
"""