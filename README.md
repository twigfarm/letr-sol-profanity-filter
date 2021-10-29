# KSL-BERT (Korean Spoken Language-BERT)

### 한국어 구어체 문장에 특화된 사전 학습 모델

<br>
공개된 한국어 BERT는 대부분 한국어 위키, 뉴스 기사, 책 등 잘 정제된 데이터를 기반으로 학습한 모델입니다. 한편, 실제로 NSMC와 같은 댓글형 데이터셋은 정제되지 않았고 구어체 특징에 신조어가 많으며, 오탈자 등 공식적인 글쓰기에서 나타나지 않는 표현들이 빈번하게 등장합니다.

KcBERT는 위와 같은 특성의 데이터셋에 적용하기 위해, 네이버 뉴스에서 댓글과 대댓글을 수집해, 토크나이저와 BERT모델을 처음부터 학습한 Pretrained BERT 모델입니다.

**KSL-BERT는 특정 도메인에 국한되지 않으며 KcBERT의 데이터에 더해 한국어 구어체의 데이터를 추가하여 처음부터 학습한 BERT 모델 기반의 Pretrained BERT 모델입니다.**

**KSL-BERT**의 Huggingface의 Transformers 라이브러리를 통해 간편히 불러와 사용할 수 있습니다. (별도의 파일 다운로드가 필요하지 않습니다.) <br>

## How to use

### Requirements

- `pytorch ~= 1.6.0`
- `transformers ~= 3.5.1`

```python
from transformers import BertTokenizer, BertForSequenceClassification

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('dobbytk/KSL-BERT')

# Load Model
model = BertModel.from_pretrained('dobbytk/KSL-BERT')
```

## Train Data & Preprocessing

### Raw Data

1. KcBERT 학습을 위한 말뭉치
2. 국립국어원, 일상 대화 말뭉치 (2020)
3. 국립국어원, 감성 분석 말뭉치 (2020)
4. 국립국어원, 구어 말뭉치
5. 트위그팜 구어 말뭉치

데이터 사이즈는 텍스트만 추출 시 약 14GB이며, 약 1억 2천만개의 문장으로 이뤄져 있습니다.

### Preprocessing

1. [KSS](https://github.com/hyunwoongko/kss) (Korean Sentence Segmentation Toolkit) 문장 쪼개기
   - 문장의 토큰 수가 BERT 모델 입력 최대 허용 범위 (512 토큰)을 넘어가는 경우를 방지하였습니다.
2. Clean 함수 적용
   - 정규표현식을 통해 한글, 영어, 특수문자, Emoji 😊 까지 학습 대상에 포함하였습니다.
   - 문장 내 중복 문자열 축약하였습니다. (같은 문자 3번 이상 반복되는 경우)
     - ex) `ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ` -> `ㅋㅋ`

```shell
pip install soynlp emoji
```

```python
import re
import emoji
from soynlp.normalizer import repeat_normalize

emojis = list({y for x in emoji.UNICODE_EMOJI.values() for y in x.keys()})
emojis = ''.join(emojis)
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
  r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
  x = pattern.sub(' ', x)
  x = url_pattern.sub('', x)
  x = x.strip()
  x = repeat_normalize(x, num_repeats=2)
  return x
```

3. Char기준 10개 이하인 문장 제거
   - 10글자 미만의 텍스트는 단일 단어로 이루어진 경우가 많아 해당 부분 제거하였습니다.
4. 감성 분석 말뭉치의 경우, 해시태그 제거
   - 감성 분석 말뭉치의 경우 SNS 특성으로 인한 해시태그(#)가 많아 해당 부분 제거하였습니다.
5. 중복 제거
   - 중복적으로 쓰인 문장을 제거하기 위해 완전일치방식으로 합쳤습니다.

이를 통해 만든 최종 학습 데이터는 13GB, 약 9700만 개 문장입니다.

## Tokenizer Train

Tokenizer는 Huggingface의 Tokenizers 라이브러리를 통해 학습을 진행했습니다.<br>
그 중 `BertWordPieceTokenizer` 를 이용하여 학습을 진행했고, Vocab Size는 `30000` 으로 진행했습니다.<br>
Tokenizer를 학습하는 것에는 데이터를 모두 사용했습니다.

## BERT Model Pretrain

- KSL-BERT Base config

```
{
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 30000,
  "model_type": "bert",
  "architectures": ["BertForMaskedLM"]
}
```

BERT Model Config 는 Base 기본 세팅값을 그대로 사용했습니다. (MLM 15% 등)<br>
Tesla V100 GPU를 이용하여 약 20일동안 진행했고, 현재 Huggigface에 공개된 모델은 250만 step을 학습한 ckpt가 업로드 되어있습니다.

- KSL-BERT Base Loss
  ![KakaoTalk_Photo_2021-10-24-22-08-51](https://user-images.githubusercontent.com/51789449/138595627-32df7639-bd4d-4da5-a9ab-4c5cdaa5ab00.png)

모델 학습 Loss는 Step에 따라 초기 500K까지 가장 빠르게 줄어들다 1M이후로는 조금씩 감소하는 것을 볼 수 있습니다.

## Example

### HuggingFace MASK LM

[HuggingFace KSL-BERT Base 모델](https://huggingface.co/dobbytk/KSL-BERT) 에서 아래와 같이 테스트 해 볼 수 있습니다.
![KakaoTalk_Image_2021-10-24-23-12-25](https://user-images.githubusercontent.com/51789449/138597972-169144db-df31-4fac-8755-b9b318ba0d9d.png)

![KakaoTalk_Image_2021-10-24-22-35-30](https://user-images.githubusercontent.com/51789449/138596570-18f19530-bbe9-41fa-be99-71a3ee0b30e8.png)

## KSL-BERT Performance

|                   | Size  | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) | **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :---------------- | :---: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :-----------------------------------: |
| KoBERT            | 351M  |       89.59        |         87.92          |       81.25        |        79.62         |           81.59           |            94.85            |         51.75 / 79.15         |                 66.21                 |
| XLM-Roberta-Base  | 1.03G |       89.03        |         86.65          |       82.80        |        80.23         |           78.45           |            93.80            |         64.70 / 88.94         |                 64.06                 |
| HanBERT           | 614M  |       90.06        |         87.70          |       82.95        |        80.32         |           82.73           |            94.72            |         78.74 / 92.02         |                 68.32                 |
| KoELECTRA-Base-v3 | 431M  |       90.63        |         88.11          |       84.45        |        82.24         |           85.53           |            95.25            |         84.83 / 93.45         |                 67.61                 |
| Soongsil-BERT     | 370M  |      **91.2**      |           -            |         -          |          -           |            76             |             94              |               -               |                **69**                 |
| KSL-BERT          | 419M  |       89.24        |           -            |         -          |          -           |             -             |              -              |               -               |                 67.30                 |

## Reference

### Github Repos

- [BERT by Google](https://github.com/google-research/bert)
- [KcBERT by Beomi](https://github.com/Beomi/KcBERT)
- [Soongsil-BERT](https://github.com/jason9693/Soongsil-BERT)
- [KSS(korean sentence segmentation)](https://github.com/hyunwoongko/kss)
- [Transformers by Huggigface](https://github.com/huggingface/transformers)
- [Tokenizers by Huggingface](https://github.com/huggingface/tokenizers)

### Papers

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
