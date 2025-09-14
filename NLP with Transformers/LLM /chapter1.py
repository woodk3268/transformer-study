# 1. 소개
# NLP 와 LLM 이해

"""
- NLP (자연어 처리):

  - 컴퓨터가 인간의 언어를 다루는 전체적인 기술 분야
  - 감정 분석, 개체명 인식, 기계 번역 등 다양한 작업 포함

- LLM (대규모 언어 모델):

  - NLP 안의 **특히 강력한 모델 그룹**
  - 방대한 데이터와 매개변수로 학습됨
  - 별도 작업별 학습이 거의 없어도 다양한 언어 작업 수행 가능

"""

# 2. 자연어 처리 및 대규모 언어 모델
# 2.1 NLP란?
"""
- NLP 작업 목록
    - 전체 문장 분류 : 리뷰의 감정 파악 ,이메일의 스팸 여부, 문장이 문법적으로 올바른지 또는 두 문장이 논리적으로 관련되어 있는지 여부 판별
    - 문장의 각 단어 분류 : 문장의 문법적 구성 요소(명사, 동사, 형용사) 또는 명명된 개체(사람, 위치, 조직) 식별
    - 텍스트 콘텐츠 생성 : 자동 생성된 텍스트로 프롬프트 완성, 마스크된 단어로 텍스트의 빈칸 채우기
    - 텍스트에서 답변 추출 : 질문과 맥락이 주어지면 맥락에 제공된 정보를 기반으로 질문에 대한 답변 추출
    - 입력 텍스트에서 새로운 문장 생성 : 텍스트를 다른 언어로 번역, 텍스트 요약
"""

# 2.2 대규모 언어모델(LLM)의 등장
"""
대규모 언어모델(LLM)은 방대한 양의 텍스트 데이터를 기반으로 훈련된 AI 모델로, 인간과 유사한 텍스트를 이해하고 생성하며,
언어 패턴을 인식하고, 특정 작업에 대한 훈련 없이도 다양한 언어 작업 수행 가능
이는 자연어 처리(NLP) 분야에서 중요한 발전을 나타냄
"""

"""
- LLM의 특징
    - 규모 : 수백만, 수십억, 심지어 수천억 개의 매개변수 포함
    - 일반 기능 : 특정 작업에 대한 훈련 없이도 여러 작업 수행 가능
    - 맥락 내 학습 : 프롬프트에 제공된 예시를 통해 학습 가능
    - 새로운 능력 : 이런 모델이 커짐에 따라 명시적으로 프로그래밍되거나 예상되지 않았던 능력이 입증됨
"""

"""
- LLM의 한계
    - 환각 : 잘못된 정보 생성
    - 진정한 이해의 부족 : 순전히 통계적 패턴에 따라 행동
    - 편향 : 훈련 데이터나 입력에 존재하는 편향 재현할 수 있음
    - 컨텍스트 창 : 컨텍스트 창이 제한되어 있음
    - 계산 리소스 : 상당한 계산 리소스가 필요함 
"""

# 3. 트랜스포머는 무엇을 할 수 있을까?

# 3.1 파이프라인 작업
# pipeline() 함수 : 트랜스포머 라이브러리에서 가장 기본적인 객체.
# 모델을 필요한 전처리 및 후처리 단계와 연결하여 어떤 텍스트든 직접 입력하면 이해할 수 있는 결과를 얻을 수 있도록 해줌.

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))

print(classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
))

# 3.2 제로샷 분류
# 레이블이 지정되지 않은 텍스트 분류
# zero-shot-classification : 분류에 사용할 레이블을 지정할 수 있으므로 사전 학습된 모델의 레이블에 의존할 필요가 없음
# 모델이 두 레이블을 사용하여 긍정 또는 부정으로 분류하는 방법을 이미 살펴보았지만, 원하는 다른 레이블 집합을 사용하여 텍스트를 분류할 수도 있음.


classifier = pipeline("zero-shot-classification")
print(classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
))

# 3.3 텍스트 생성
# 프롬프트를 입력하면 모델이 나머지 텍스트를 생성하여 프롬프트를 자동 완성
# 텍스트 생성에는 무작위성이 포함됨
generator = pipeline("text-generation")
print(generator("In this course, we will teach you how to"))


# 추가. 파이프라인에서 허브의 모든 모델 사용
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
print(generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
    do_sample=True,
))

# 3.4 fill mask
# 주어진 텍스트의 빈칸을 채움
# top_k : 표시할 가능성의 개수 제어
unmasker = pipeline("fill-mask")
print(unmasker("This course will teach you all about <mask> models.", top_k=2))

# 3.5 개체명 인식
# 모델이 입력 테스트의 어떤 부분이 사람, 위치, 조직 등의 개체에 해당하는지 찾아내는 작업
# 파이프라인 생성 함수에 옵션을 전달하여 (grouped_entities=True) 파이프라인에 동일한 엔티티에 해당하는 문장 부분들을 다시 그룹화하도록 지시
# 모델은 "Hugging"과 "Face"라는 이름이 여러 단어로 구성되어 있음에도 불구하고 하나의 조직으로 올바르게 그룹화
ner = pipeline("ner", grouped_entities=True)
print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

# 3.6 질의응답
# 제공된 컨텍스트에서 정보 추출
question_answer = pipeline("question-answering")
print(question_answer(
    question = "어디에서 일하나요?",
    context = "제 이름은 Sylvain이고 브루클린의 Hugging Face에서 일합니다."
))

# 3.7 요약
# 본문에 언급된 중요한 부분을 모두 유지하면서 본문을 더 짧은 텍스트로 줄이는 작업

summarizer = pipeline("summarization")
print(summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
))

# 3.8 번역
# 작업 이름에 언어 쌍을 입력하면 기본 모델을 사용할 수 있지만, 가장 쉬운 방법은 모델 허브에서 사용할 모델 선택
# 여기서는 프랑스어에서 영어로 번역
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
print(translator("Ce cours est produit par Hugging Face."))

# 3.9 이미지 및 오디오 파이프라인
# 3.9.1 이미지 분류

image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)

# 3.9.10 자동 음성 인식
transcriber = pipeline(
    task="automatic-speech-recognition", model="openai/whisper-large-v3"
)
result = transcriber(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
)
print(result)

# 4. 트랜스포머 작동 원리
# 4.1 언어 모델
"""
- 트랜스포머 모델(GPT, BERT, T5 등)은 언어 모델로 학습
- 즉, 이 모델들은 대량의 원시 텍스트를 기반으로 자기 지도 학습 방식으로 학습됨.
- 자기 지도 학습은 모델의 입력값을 바탕으로 목표가 자동으로 계산되는 학습 유형
- 즉, 사람이 데이터에 레이블을 지정할 필요가 없음
- 이러한 유형의 모델은 학습된 언어에 대한 통계적 이해를 개발하지만, 특정 실무 작업에는 그다지 유용하지 않음.
- 따라서 일반적인 사전 학습된 모델은 전이 학습 또는 미세 조정이라는 과정을 거침.
- 이 과정에서, 모델은 주어진 작업에 대해 지도학습 방식 ,즉 사람이 주석을 단 레이블을 사용하여 미세조정됨

예) - 인과적 언어 모델링 : 이전 n개의 단어를 읽고 문장의 다음 단어를 예측. 출력은 과거와 현재 입력에 따라 달라지지만, 미래 입력에는 영향 받지 않음.
    - 마스크된 언어 모델링 : 모델이 문장에서 마스크된 단어를 예측하는 방식
"""

# 4.2 대형 모델
"""
- 더 나은 성능을 달성하기 위한 일반적인 전략은 모델의 크기와 모델을 사전에 학습하는 데이터 양을 늘리는 것
- 이는 시간과 컴퓨팅 리소스 측면에서 매우 큰 비용을 초래
- 언어 모델을 공유하는 것이 가장 중요한 이유는 훈련된 가중치를 공유하고 이미 훈련된 가중치를 기반으로 구축하면 전반적인 컴퓨팅 비용이 감소하기 때문
"""

# 4.2.1 전이 학습
"""
- 미세 조정은 모델이 사전학습된 후에 수행되는 학습
- 미세 조정을 수행하려면 먼저 사전 학습된 언어 모델을 확보한 다음, 작업에 특화된 데이터셋을 사용하여 추가학습 수행
"""

# 5. 트랜스포머의 일반적인 아키텍처
"""
- 인코더 : 입력을 받아 그 표현(특징)을 구축. 즉, 모델은 입력으로부터 이해를 얻도록 최적화됨
- 디코더 : 인코더의 표현(특징)을 다른 입력과 함께 사용하여 목표 시퀀스 생성. 즉, 모델이 출력 생성에 최적화되어 있음을 의미.

- 인코더 전용 모델 : 문장 분류 및 명명된 엔티티 인식과 같이 입력에 대한 이해가 필요한 작업에 적합
- 디코더 전용 모델 : 텍스트 생성과 같은 생성 작업에 적합
- 인코더-디코더 모델 또는 시퀀스-투-시퀀스 모델 : 번역이나 요약과 같이 입력이 필요한 생성 작업에 적합
"""

# 6. attention layer
"""
 1. **주의 레이어(Attention Layer)의 개념**

* 트랜스포머 모델의 핵심은 **주의(attention)** 라는 메커니즘
* 이름 그대로, 모델이 입력 문장을 처리할 때 **특정 단어에 더 집중(주의)** 하고, 덜 중요한 단어는 무시하도록 도움

---

 2. **예시: 번역**

영어 → 프랑스어 번역
입력: `"You like this course"`

* **"like" 번역할 때**

  * 프랑스어 동사 "aimer"는 주어(You, He, They…)에 따라 변형됨
  * 따라서 "like"를 번역하려면 **주어 "You"** 에 주의를 기울여야 함.
  * "this"나 "course" 같은 다른 단어는 중요하지 않음.

* **"this" 번역할 때**

  * 프랑스어에서 "this"는 명사의 성(남성/여성)에 따라 달라짐.

    * 남성: *ce cours*
    * 여성: *cette classe*
  * 따라서 "this"를 올바르게 번역하려면 **뒤의 "course"** 단어에 주의를 기울여야 함

즉, 모델은 단순히 순서대로 단어를 보지 않고, **필요한 단어에 selective attention** 을 함.

---

 3. **왜 중요한가?**

* 단어의 의미는 고립되어 있지 않고 **맥락(context)** 에 따라 달라짐
* Attention은 이 맥락을 학습해서, 단어 간 관계를 찾아내고 활용함.
* 복잡한 문장일수록, 멀리 떨어진 단어와의 관계가 중요해짐. → 이걸 RNN은 잘 못하지만, Transformer는 Attention 덕분에 잘 처리함


"""
# 7. 원래의 구조
"""

 1. 트랜스포머의 기본 목적

* 트랜스포머는 원래 **기계 번역**(예: 영어 → 프랑스어)을 위해 설계됨.
* 따라서 구조가 두 부분으로 나뉨:

  * **인코더(Encoder):** 입력 문장(영어)을 읽고, 의미를 압축해 표현.
  * **디코더(Decoder):** 인코더 출력을 보고 목표 문장(프랑스어)을 한 단어씩 생성.

---

2. 인코더의 어텐션

* 인코더는 입력 문장의 **모든 단어에 동시에 주의(attention)** 를 기울일 수 있음.
* 예: `"You like this course"`라는 영어 문장을 입력받으면, 인코더는 `"You"`, `"like"`, `"this"`, `"course"` 사이의 관계를 모두 고려해 문맥 정보를 만든 뒤 디코더에 전달.

---

3. 디코더의 어텐션

디코더에는 **두 종류의 어텐션**이 있음.

1. **첫 번째 어텐션 (Masked Self-Attention)**

   * 디코더는 목표 문장을 **순차적으로** 생성.
   * 예: `"Tu aimes ce cours"`를 만들 때, 네 번째 단어 `"cours"`를 예측할 때는 앞 단어 `"Tu aimes ce"`까지만 보고 예측해야 함.
   * 미래 단어(`"cours"`)는 가려져 있어야 함.
   * 그래서 "마스크(mask)"를 사용해 **앞 단어까지만 접근** 가능하게 만듦.

2. **두 번째 어텐션 (Cross-Attention)**

   * 디코더는 인코더의 전체 출력(즉, 원문 전체)을 참고함.
   * 이렇게 해야 번역할 때 단어 순서 차이나 문법 구조 차이를 반영 가능.
   * 예: 영어는 `"like"`가 동사지만, 프랑스어는 주어에 따라 `"aime"`, `"aimes"`, `"aimons"` 등으로 바뀌므로 인코더 전체 입력을 활용해야 함.

---

4. 학습 시 처리 방식

* 학습 중에는 디코더에 **정답 문장 전체**를 입력해도 됨.
* 하지만 마스크 때문에, 예를 들어 4번째 단어를 예측할 때는 앞의 1\~3번째 단어까지만 쓸 수 있음.
* 이렇게 해서 모델이 단순 복사하지 않고, 실제로 "다음 단어 예측"을 학습하게 만듦.

---

5. 어텐션 마스크(Attention Mask)

* 마스크는 크게 두 가지 용도로 쓰임:

  1. **디코더 마스크(Masked Attention):** 미래 단어를 못 보게 차단.
  2. **패딩 마스크(Padding Mask):** 문장을 배치(batch)로 묶을 때 길이가 다른 경우 패딩(`PAD`) 토큰이 들어가는데, 모델이 이 토큰에 쓸데없이 주의하지 않도록 무시하게 만듦.


"""