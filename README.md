# 2026summercamp
# 🎬 YouTube 자막 요약 및 관련 영상 추천 웹앱

사용자가 YouTube 영상 링크를 입력하면 영상의 자막을 분석하여 핵심 내용을 요약하고, 관련 영상을 추천하는 Streamlit 웹 애플리케이션입니다.

추천 과정에서는 영상 텍스트를 **TF-IDF로 벡터화**한 뒤 **코사인 유사도**를 계산합니다. 이후 조회수, 좋아요 수, 업로드 후 경과 일수를 함께 반영하여 최종 추천 순위를 결정합니다.

분석 결과는 CSV 파일로 내려받아 **Orange3에서 유사도와 군집을 시각화**할 수 있습니다.

---

## 📌 프로젝트 목적

이 프로젝트의 연구 질문은 다음과 같습니다.

> 텍스트 유사도와 영상 인기도를 결합한 추천 방식은 조회수만 사용하는 방식보다 입력 영상과 관련성이 높은 영상을 추천할 수 있는가?

이 프로젝트를 통해 다음 개념을 탐구합니다.

- 자연어 처리
- TF-IDF
- 코사인 유사도
- 추출 요약
- 추천 시스템
- 데이터 정규화
- 데이터 시각화
- 군집 분석

---

## ✨ 주요 기능

- YouTube URL에서 영상 ID 추출
- 한국어 및 영어 자막 수집
- 자막 텍스트 전처리
- TF-IDF 기반 핵심 문장 추출
- 핵심 키워드 추출
- YouTube Data API를 이용한 관련 영상 검색
- 입력 영상과 후보 영상의 코사인 유사도 계산
- 일평균 조회수와 좋아요 수를 반영한 추천
- 추천 결과 그래프 표시
- 추천 결과 CSV 다운로드
- Orange3 군집 분석용 TF-IDF CSV 다운로드

---

## ⚙️ 작동 원리

### 1. YouTube 영상 입력

사용자가 분석할 YouTube 영상 URL을 입력합니다.

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

웹앱은 URL에서 영상 ID를 추출합니다.

### 2. 자막 수집

`youtube-transcript-api`를 사용하여 영상 자막을 가져옵니다.

자막은 다음 순서로 선택합니다.

1. 한국어 자막
2. 영어 자막
3. 그 외 사용 가능한 자막

자막이 없는 영상은 분석할 수 없습니다.

### 3. 자막 전처리

수집한 자막에서 다음 요소를 제거하거나 정리합니다.

- 불필요한 공백
- URL
- 음악, 박수, 웃음 등의 자막 표시
- 일부 특수문자
- 한국어 및 영어 불용어

### 4. TF-IDF 기반 요약

자막을 문장 단위로 나누고 각 문장에 대해 TF-IDF 점수를 계산합니다.

TF-IDF는 문서 안에서 특정 단어가 얼마나 중요한지를 나타내는 값입니다.

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

TF-IDF 점수가 높은 단어를 많이 포함한 문장을 핵심 문장으로 선택합니다.

이 방식은 새로운 문장을 만드는 생성 요약이 아니라, 기존 자막에서 중요한 문장을 선택하는 **추출 요약** 방식입니다.

### 5. 관련 영상 검색

추출된 핵심 키워드를 검색어로 사용하여 YouTube Data API에서 관련 영상 후보를 가져옵니다.

후보 영상의 다음 정보를 수집합니다.

- 제목
- 설명
- 채널명
- 조회수
- 좋아요 수
- 업로드 날짜
- 썸네일

### 6. 코사인 유사도 계산

입력 영상의 자막과 후보 영상의 제목·설명을 TF-IDF 벡터로 변환합니다.

두 벡터 사이의 코사인 유사도를 계산하여 내용의 관련성을 측정합니다.

\[
CosineSimilarity(A,B)
=
\frac{A\cdot B}
{\|A\|\|B\|}
\]

코사인 유사도가 1에 가까울수록 두 텍스트의 내용이 유사하다고 볼 수 있습니다.

### 7. 인기도 보정

오래된 영상이 조회수에서 지나치게 유리해지는 문제를 줄이기 위해 일평균 조회수를 사용합니다.

\[
DailyViews
=
\frac{Views}
{DaysSinceUpload}
\]

조회수와 좋아요 수는 영상마다 차이가 매우 크므로 로그 변환과 최소-최대 정규화를 적용합니다.

\[
x'=\log(1+x)
\]

### 8. 최종 추천 점수

기본 추천 점수는 다음과 같이 계산합니다.

\[
Score
=
0.7S+0.2V+0.1L
\]

- \(S\): 코사인 유사도
- \(V\): 정규화된 일평균 조회수
- \(L\): 정규화된 좋아요 수

가중치는 Streamlit 사이드바에서 직접 조절할 수 있습니다.

---

## 🛠️ 사용 기술

| 구분 | 사용 기술 |
|---|---|
| 개발 언어 | Python |
| 웹 프레임워크 | Streamlit |
| 데이터 처리 | Pandas, NumPy |
| 자연어 처리 | TF-IDF |
| 유사도 계산 | Cosine Similarity |
| 머신러닝 라이브러리 | Scikit-learn |
| 영상 정보 수집 | YouTube Data API v3 |
| 자막 수집 | youtube-transcript-api |
| 버전 관리 | GitHub |
| 데이터 시각화 | Streamlit, Orange3 |

---

## 📁 프로젝트 구조

```text
youtube-summary-recommender/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

각 파일의 역할은 다음과 같습니다.

| 파일 | 역할 |
|---|---|
| `app.py` | 웹앱 전체 코드 |
| `requirements.txt` | 필요한 Python 패키지 목록 |
| `README.md` | 프로젝트 설명서 |
| `.gitignore` | API 키와 불필요한 파일의 업로드 방지 |

---

## 🔑 YouTube Data API 키 발급

영상 검색, 제목, 설명, 조회수, 좋아요 수를 가져오려면 YouTube Data API 키가 필요합니다.

1. [Google Cloud Console](https://console.cloud.google.com/)에 접속합니다.
2. Google 계정으로 로그인합니다.
3. 새 프로젝트를 생성합니다.
4. `API 및 서비스` 메뉴로 이동합니다.
5. `라이브러리`를 선택합니다.
6. `YouTube Data API v3`를 검색합니다.
7. API를 활성화합니다.
8. `사용자 인증 정보`에서 API 키를 생성합니다.
9. 필요하면 API 키 사용 제한을 설정합니다.

> API 키를 `app.py` 또는 README에 직접 입력하지 마세요.

---

## 🚀 Streamlit Community Cloud 배포

### 1. GitHub 저장소 생성

GitHub에서 새 저장소를 만든 뒤 다음 파일을 업로드합니다.

```text
app.py
requirements.txt
README.md
.gitignore
```

### 2. Streamlit Community Cloud 접속

[Streamlit Community Cloud](https://share.streamlit.io/)에 접속하여 GitHub 계정으로 로그인합니다.

### 3. 앱 생성

다음 항목을 설정합니다.

- Repository: 프로젝트 GitHub 저장소
- Branch: `main`
- Main file path: `app.py`

### 4. API 키 등록

Streamlit 앱 설정의 `Secrets`에 다음 내용을 입력합니다.

```toml
YOUTUBE_API_KEY = "본인의_실제_API_키"
```

API 키는 GitHub에 올리지 않고 Streamlit Secrets에만 저장합니다.

### 5. 배포

`Deploy` 버튼을 누르면 필요한 패키지가 자동으로 설치되고 웹앱이 실행됩니다.

---

## 💻 로컬 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/본인아이디/저장소이름.git
cd 저장소이름
```

### 2. 가상환경 생성

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS 또는 Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. API 키 파일 생성

프로젝트 폴더 안에 다음 구조로 파일을 만듭니다.

```text
.streamlit/
└── secrets.toml
```

`secrets.toml`에 API 키를 작성합니다.

```toml
YOUTUBE_API_KEY = "본인의_실제_API_키"
```

### 5. 앱 실행

```bash
streamlit run app.py
```

---

## 📦 requirements.txt

```txt
streamlit>=1.36,<2.0
pandas>=2.2,<3.0
numpy>=1.26,<3.0
scikit-learn>=1.5,<2.0
google-api-python-client>=2.137,<3.0
youtube-transcript-api>=1.0,<2.0
```

---

## 🔒 .gitignore

```gitignore
.streamlit/secrets.toml
.env
.venv/
venv/
__pycache__/
*.pyc
```

`.gitignore`는 API 키가 저장된 파일과 실행 과정에서 생성되는 불필요한 파일이 GitHub에 업로드되지 않도록 합니다.

---

## 📊 Orange3 분석 방법

웹앱에서 다음 파일을 다운로드할 수 있습니다.

- `youtube_recommendations.csv`
- `orange_tfidf_features.csv`

### 추천 결과 CSV

다음과 같은 정보를 포함합니다.

- 영상 ID
- 영상 제목
- 채널명
- 조회수
- 좋아요 수
- 일평균 조회수
- 코사인 유사도
- 최종 추천 점수

### TF-IDF 특징 CSV

후보 영상별 TF-IDF 특징값을 포함합니다. 내용 기반 군집 분석에 활용할 수 있습니다.

### 권장 Orange3 워크플로

```text
File
→ Select Columns
→ Normalize
→ Distances
→ Hierarchical Clustering
→ MDS 또는 t-SNE
→ Scatter Plot
```

`Select Columns`에서는 다음과 같이 설정하는 것이 좋습니다.

- `tfidf_`로 시작하는 열: Feature
- 제목, 채널명, URL: Meta
- 코사인 유사도와 추천 점수: Feature 또는 Meta

이를 통해 비슷한 내용을 가진 영상들이 가까운 위치에 배치되는지 확인할 수 있습니다.

---

## 🧪 평가 방법

다음 세 추천 방식을 비교할 수 있습니다.

- 모델 A: 조회수만 사용
- 모델 B: 코사인 유사도만 사용
- 모델 C: 코사인 유사도, 일평균 조회수, 좋아요 수를 결합

추천 결과의 관련성을 다음 기준으로 평가합니다.

| 평가 | 점수 |
|---|---:|
| 입력 영상과 관련 있음 | 2점 |
| 어느 정도 관련 있음 | 1점 |
| 관련 없음 | 0점 |

각 모델이 추천한 상위 5개 영상의 평균 점수를 비교하여 결합 추천 방식의 효과를 분석할 수 있습니다.

---

## ⚠️ 한계점

- 모든 YouTube 영상에 자막이 존재하는 것은 아닙니다.
- 자동 생성 자막에는 오류가 포함될 수 있습니다.
- TF-IDF는 동의어와 문맥적 의미를 충분히 이해하지 못합니다.
- 후보 영상은 처리 속도를 고려하여 제목과 설명으로 비교합니다.
- 제목과 설명이 영상의 실제 내용을 완전히 나타내지 않을 수 있습니다.
- 좋아요 수가 비공개인 경우 0으로 처리될 수 있습니다.
- YouTube Data API에는 일일 할당량 제한이 있습니다.
- 서로 다른 언어의 영상은 정확한 유사도 비교가 어려울 수 있습니다.

---

## 🔧 향후 개선 방향

- 후보 영상의 자막까지 수집하여 비교
- 형태소 분석기를 이용한 한국어 전처리 개선
- Sentence-BERT 기반 의미 유사도 추가
- 영상 길이를 반영한 요약 문장 수 자동 설정
- 자막이 없는 영상의 제목·설명 기반 대체 분석
- 추천 결과에 대한 사용자 평가 기능 추가
- 조회수 기반 모델과 결합 모델의 비교 실험 자동화
- Orange3 분석 결과를 연구 보고서에 반영

---

## 📚 연구 주제

**TF-IDF와 코사인 유사도를 활용한 YouTube 자막 요약 및 콘텐츠·인기도 기반 영상 추천 시스템**

---

## 👨‍💻 제작

- 학교: 당곡고등학교
- 제작자: 이름 입력
- 프로젝트 유형: 정보·인공지능·데이터 분석 프로젝트
