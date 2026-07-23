import os
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="YouTube TF-IDF 자막 분석 및 추천",
    page_icon="🎬",
    layout="wide",
)


# =========================================================
# 추천 모델의 고정 설정
# 사용자가 임의로 바꾸지 않도록 코드 내부에 고정합니다.
# =========================================================

TEXT_WEIGHT = 0.90
POPULARITY_WEIGHT = 0.10

TITLE_SIMILARITY_WEIGHT = 0.30
CONTENT_SIMILARITY_WEIGHT = 0.70

MINIMUM_SIMILARITY = 0.15
MAX_TFIDF_FEATURES = 1000
MAX_TRANSCRIPT_CHARACTERS = 20000


# =========================================================
# 불용어
# =========================================================

KOREAN_STOP_WORDS = {
    "이", "그", "저", "것", "수", "등", "들", "및", "더",
    "를", "을", "은", "는", "가", "에", "의", "와", "과",
    "로", "으로", "에서", "에게", "한", "하다", "합니다",
    "있다", "있는", "있습니다", "없다", "그리고", "하지만",
    "또한", "때문", "대한", "대해", "통해", "영상", "오늘",
    "여러분", "제가", "저희", "우리", "이제", "정말", "좀",
    "잘", "거", "게", "건", "입니다", "됩니다", "하면",
    "해서", "하는", "하고", "같은", "이런", "그런", "그래서",
    "하는데", "합니다", "됩니다", "있는데", "보면", "한번",
    "아니", "그러면", "그냥", "지금", "여기", "이렇게",
    "저렇게", "뭐", "또", "제가", "아주", "많이",
}


# =========================================================
# 텍스트 전처리
# =========================================================

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*(음악|박수|웃음)[^\)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_mixed_text(text):
    """
    한국어와 영어를 함께 처리하는 토크나이저입니다.

    TF-IDF 벡터라이저는 이 함수로 텍스트를 단어 목록으로 변환합니다.
    """
    text = clean_text(text).lower()

    tokens = re.findall(
        r"[가-힣]{2,}|[a-zA-Z][a-zA-Z0-9_-]{1,}",
        text,
    )

    return [
        token
        for token in tokens
        if token not in KOREAN_STOP_WORDS
        and token not in ENGLISH_STOP_WORDS
        and len(token) >= 2
    ]


def split_sentences(text):
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+|(?<=[다요죠])\s+(?=[가-힣A-Z0-9])",
        text,
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if 20 <= len(sentence.strip()) <= 600
    ]

    # 자동 자막에 문장부호가 거의 없는 경우 일정 길이로 분할합니다.
    if len(sentences) <= 1 and len(text) > 500:
        chunk_size = 300

        sentences = [
            text[index:index + chunk_size].strip()
            for index in range(0, len(text), chunk_size)
            if len(text[index:index + chunk_size].strip()) >= 20
        ]

    return sentences


def limit_text(text, maximum_characters=MAX_TRANSCRIPT_CHARACTERS):
    """
    지나치게 긴 자막이 서버 메모리를 과도하게 사용하지 않도록 제한합니다.
    """
    text = clean_text(text)

    if len(text) <= maximum_characters:
        return text

    return text[:maximum_characters]


# =========================================================
# YouTube URL 및 API 함수
# =========================================================

def extract_video_id(url):
    if not url:
        return None

    url = url.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().replace("www.", "")

    if hostname == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if len(candidate) == 11 else None

    if hostname in {
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]

            if candidate and len(candidate) == 11:
                return candidate

        parts = parsed.path.strip("/").split("/")

        if len(parts) >= 2 and parts[0] in {
            "shorts",
            "embed",
            "live",
        }:
            candidate = parts[1]
            return candidate if len(candidate) == 11 else None

    match = re.search(
        r"(?:v=|youtu\.be/|shorts/|embed/|live/)"
        r"([A-Za-z0-9_-]{11})",
        url,
    )

    return match.group(1) if match else None


def get_api_key():
    try:
        key = st.secrets["YOUTUBE_API_KEY"]

        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass

    return os.getenv("YOUTUBE_API_KEY", "")


def get_youtube_client(api_key):
    return build(
        "youtube",
        "v3",
        developerKey=api_key,
        cache_discovery=False,
    )


def calculate_days_since_upload(published_at):
    if not published_at:
        return 1

    published_date = datetime.fromisoformat(
        published_at.replace("Z", "+00:00")
    )

    days = (
        datetime.now(timezone.utc) - published_date
    ).days

    return max(days, 1)


def get_thumbnail(snippet):
    thumbnails = snippet.get("thumbnails", {})

    return (
        thumbnails.get("high", {}).get("url")
        or thumbnails.get("medium", {}).get("url")
        or thumbnails.get("default", {}).get("url")
        or ""
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_transcript(video_id):
    """
    한국어 → 영어 → 사용 가능한 언어 순으로 자막을 가져옵니다.
    """
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(
            video_id,
            languages=["ko", "en", "en-US", "en-GB"],
        )

        text = " ".join(
            snippet.text.replace("\n", " ")
            for snippet in list(fetched)
        )

        return {
            "success": True,
            "text": limit_text(text),
            "language": getattr(
                fetched,
                "language_code",
                "unknown",
            ),
        }

    except Exception:
        try:
            transcript_list = api.list(video_id)
            transcripts = list(transcript_list)

            if not transcripts:
                return {
                    "success": False,
                    "text": "",
                    "language": "none",
                }

            selected = None

            for transcript in transcripts:
                if transcript.language_code.startswith("ko"):
                    selected = transcript
                    break

            if selected is None:
                for transcript in transcripts:
                    if transcript.language_code.startswith("en"):
                        selected = transcript
                        break

            if selected is None:
                selected = transcripts[0]

            fetched = selected.fetch()

            text = " ".join(
                snippet.text.replace("\n", " ")
                for snippet in list(fetched)
            )

            return {
                "success": True,
                "text": limit_text(text),
                "language": selected.language_code,
            }

        except Exception:
            return {
                "success": False,
                "text": "",
                "language": "none",
            }


@st.cache_data(ttl=3600, show_spinner=False)
def get_video_details(video_id, api_key):
    youtube = get_youtube_client(api_key)

    response = (
        youtube.videos()
        .list(
            part="snippet,statistics",
            id=video_id,
        )
        .execute()
    )

    items = response.get("items", [])

    if not items:
        return None

    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})

    views = int(statistics.get("viewCount", 0))
    likes = int(statistics.get("likeCount", 0))
    published_at = snippet.get("publishedAt", "")
    days = calculate_days_since_upload(published_at)

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": published_at,
        "views": views,
        "likes": likes,
        "daily_views": views / days,
        "thumbnail": get_thumbnail(snippet),
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def search_candidate_videos(query, api_key, max_results):
    youtube = get_youtube_client(api_key)

    response = (
        youtube.search()
        .list(
            part="snippet",
            q=query,
            type="video",
            maxResults=min(max_results, 50),
            order="relevance",
            safeSearch="moderate",
        )
        .execute()
    )

    video_ids = [
        item["id"]["videoId"]
        for item in response.get("items", [])
        if item.get("id", {}).get("videoId")
    ]

    if not video_ids:
        return []

    detail_response = (
        youtube.videos()
        .list(
            part="snippet,statistics",
            id=",".join(video_ids),
        )
        .execute()
    )

    candidates = []

    for item in detail_response.get("items", []):
        video_id = item.get("id", "")
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})

        views = int(statistics.get("viewCount", 0))
        likes = int(statistics.get("likeCount", 0))
        published_at = snippet.get("publishedAt", "")
        days = calculate_days_since_upload(published_at)

        candidates.append(
            {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": published_at,
                "views": views,
                "likes": likes,
                "daily_views": views / days,
                "thumbnail": get_thumbnail(snippet),
                "url": (
                    "https://www.youtube.com/watch?v="
                    f"{video_id}"
                ),
            }
        )

    order = {
        video_id: index
        for index, video_id in enumerate(video_ids)
    }

    candidates.sort(
        key=lambda video: order.get(
            video["video_id"],
            len(video_ids),
        )
    )

    return candidates


# =========================================================
# TF-IDF 요약 및 분석
# =========================================================

def summarize_with_tfidf(
    transcript_text,
    sentence_count=5,
    keyword_count=20,
):
    """
    문장을 TF-IDF 벡터로 만든 뒤 문장별 TF-IDF 합계를 계산합니다.

    문장 중요도:
        sentence_score = 해당 문장의 모든 단어 TF-IDF 합계
    """
    sentences = split_sentences(transcript_text)

    if not sentences:
        raise ValueError("분석할 수 있는 자막 문장이 없습니다.")

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=MAX_TFIDF_FEATURES,
    )

    matrix = vectorizer.fit_transform(sentences)

    feature_names = vectorizer.get_feature_names_out()

    sentence_scores = np.asarray(
        matrix.sum(axis=1)
    ).ravel()

    top_count = min(sentence_count, len(sentences))

    important_indices = np.argsort(
        sentence_scores
    )[::-1][:top_count]

    ordered_indices = sorted(important_indices.tolist())

    summary = " ".join(
        sentences[index]
        for index in ordered_indices
    )

    ranked_sentences = pd.DataFrame(
        [
            {
                "순위": rank + 1,
                "원래 문장 순서": int(index + 1),
                "TF-IDF 문장 점수": round(
                    float(sentence_scores[index]),
                    4,
                ),
                "문장": sentences[index],
            }
            for rank, index in enumerate(important_indices)
        ]
    )

    total_term_scores = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    keyword_indices = np.argsort(
        total_term_scores
    )[::-1][:keyword_count]

    keywords = [
        feature_names[index]
        for index in keyword_indices
    ]

    keyword_df = pd.DataFrame(
        {
            "순위": range(1, len(keyword_indices) + 1),
            "단어 또는 구": [
                feature_names[index]
                for index in keyword_indices
            ],
            "전체 TF-IDF 점수": [
                round(float(total_term_scores[index]), 6)
                for index in keyword_indices
            ],
        }
    )

    return {
        "summary": summary,
        "keywords": keywords,
        "sentence_scores": ranked_sentences,
        "keyword_scores": keyword_df,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def create_tfidf_explanation_table(
    source_text,
    comparison_text,
    maximum_terms=30,
):
    """
    기준 영상과 비교 영상에 대해 TF, IDF, TF-IDF 값을 계산하여
    화면에 표시할 데이터프레임을 만듭니다.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        sublinear_tf=False,
        use_idf=True,
        smooth_idf=True,
        norm=None,
        max_features=MAX_TFIDF_FEATURES,
    )

    tfidf_matrix = vectorizer.fit_transform(
        [source_text, comparison_text]
    )

    terms = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_

    source_tokens = tokenize_mixed_text(source_text)
    comparison_tokens = tokenize_mixed_text(comparison_text)

    source_count = {
        term: source_tokens.count(term)
        for term in set(source_tokens)
    }

    comparison_count = {
        term: comparison_tokens.count(term)
        for term in set(comparison_tokens)
    }

    source_total = max(len(source_tokens), 1)
    comparison_total = max(len(comparison_tokens), 1)

    source_tfidf = tfidf_matrix[0].toarray().ravel()
    comparison_tfidf = tfidf_matrix[1].toarray().ravel()

    rows = []

    for index, term in enumerate(terms):
        rows.append(
            {
                "단어": term,
                "기준 영상 TF": round(
                    source_count.get(term, 0) / source_total,
                    6,
                ),
                "비교 영상 TF": round(
                    comparison_count.get(term, 0)
                    / comparison_total,
                    6,
                ),
                "IDF": round(float(idf_values[index]), 6),
                "기준 영상 TF-IDF": round(
                    float(source_tfidf[index]),
                    6,
                ),
                "비교 영상 TF-IDF": round(
                    float(comparison_tfidf[index]),
                    6,
                ),
                "TF-IDF 합계": round(
                    float(
                        source_tfidf[index]
                        + comparison_tfidf[index]
                    ),
                    6,
                ),
            }
        )

    dataframe = pd.DataFrame(rows)

    return dataframe.sort_values(
        "TF-IDF 합계",
        ascending=False,
    ).head(maximum_terms)


# =========================================================
# 추천 모델
# =========================================================

def normalize_log_values(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    logged = np.log1p(values).reshape(-1, 1)

    if len(values) == 1 or np.allclose(logged, logged[0]):
        return np.zeros(len(values))

    return MinMaxScaler().fit_transform(logged).ravel()


def calculate_similarity(
    source_title,
    source_content,
    candidates,
):
    """
    제목과 본문을 별도로 TF-IDF 변환합니다.

    최종 텍스트 유사도:
        0.3 × 제목 코사인 유사도
        + 0.7 × 자막/본문 코사인 유사도
    """
    candidate_titles = [
        candidate["title"]
        for candidate in candidates
    ]

    candidate_contents = [
        candidate["comparison_content"]
        for candidate in candidates
    ]

    # 제목 TF-IDF
    title_vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=500,
    )

    title_matrix = title_vectorizer.fit_transform(
        [source_title] + candidate_titles
    )

    title_similarities = cosine_similarity(
        title_matrix[0:1],
        title_matrix[1:],
    ).ravel()

    # 자막 또는 설명 TF-IDF
    content_vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=MAX_TFIDF_FEATURES,
    )

    content_matrix = content_vectorizer.fit_transform(
        [source_content] + candidate_contents
    )

    content_similarities = cosine_similarity(
        content_matrix[0:1],
        content_matrix[1:],
    ).ravel()

    combined_similarities = (
        TITLE_SIMILARITY_WEIGHT * title_similarities
        + CONTENT_SIMILARITY_WEIGHT * content_similarities
    )

    return {
        "title_similarities": title_similarities,
        "content_similarities": content_similarities,
        "combined_similarities": combined_similarities,
        "content_vectorizer": content_vectorizer,
        "content_matrix": content_matrix,
    }


def recommend_videos(
    source_title,
    source_content,
    candidates,
):
    similarity_result = calculate_similarity(
        source_title,
        source_content,
        candidates,
    )

    title_similarities = similarity_result[
        "title_similarities"
    ]

    content_similarities = similarity_result[
        "content_similarities"
    ]

    combined_similarities = similarity_result[
        "combined_similarities"
    ]

    selected_indices = [
        index
        for index, similarity in enumerate(
            combined_similarities
        )
        if similarity >= MINIMUM_SIMILARITY
    ]

    fallback_used = False

    if not selected_indices:
        fallback_used = True

        selected_indices = np.argsort(
            combined_similarities
        )[::-1][:min(5, len(candidates))].tolist()

    selected_candidates = [
        candidates[index]
        for index in selected_indices
    ]

    selected_title_similarities = title_similarities[
        selected_indices
    ]

    selected_content_similarities = content_similarities[
        selected_indices
    ]

    selected_combined_similarities = combined_similarities[
        selected_indices
    ]

    daily_views = [
        candidate["daily_views"]
        for candidate in selected_candidates
    ]

    likes = [
        candidate["likes"]
        for candidate in selected_candidates
    ]

    normalized_views = normalize_log_values(daily_views)
    normalized_likes = normalize_log_values(likes)

    popularity_scores = (
        0.6 * normalized_views
        + 0.4 * normalized_likes
    )

    final_scores = (
        TEXT_WEIGHT * selected_combined_similarities
        + POPULARITY_WEIGHT * popularity_scores
    )

    rows = []

    for index, candidate in enumerate(selected_candidates):
        row = candidate.copy()

        row["title_similarity"] = float(
            selected_title_similarities[index]
        )

        row["content_similarity"] = float(
            selected_content_similarities[index]
        )

        row["similarity"] = float(
            selected_combined_similarities[index]
        )

        row["popularity_score"] = float(
            popularity_scores[index]
        )

        row["final_score"] = float(final_scores[index])
        row["fallback_used"] = fallback_used

        rows.append(row)

    results = pd.DataFrame(rows)

    # 관련성을 우선하기 위해 유사도를 1차 정렬 기준으로 사용합니다.
    results = results.sort_values(
        by=["similarity", "final_score"],
        ascending=False,
    ).reset_index(drop=True)

    content_matrix = similarity_result["content_matrix"]
    content_vectorizer = similarity_result["content_vectorizer"]

    selected_vectors = content_matrix[1:][selected_indices]
    feature_names = content_vectorizer.get_feature_names_out()

    feature_columns = [
        f"tfidf_{index:04d}_{term.replace(' ', '_')}"
        for index, term in enumerate(feature_names)
    ]

    feature_df = pd.DataFrame(
        selected_vectors.toarray(),
        columns=feature_columns,
    )

    orange_metadata = results[
        [
            "video_id",
            "title",
            "channel_title",
            "url",
            "transcript_available",
            "title_similarity",
            "content_similarity",
            "similarity",
            "views",
            "likes",
            "daily_views",
            "popularity_score",
            "final_score",
        ]
    ].copy()

    orange_df = pd.concat(
        [
            orange_metadata.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return {
        "results": results,
        "orange_features": orange_df,
    }


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# =========================================================
# 화면 구성
# =========================================================

st.title("🎬 YouTube TF-IDF 자막 분석 및 관련 영상 추천")
    if st.sidebar.button(
    "🗑️ 캐시 삭제",
    use_container_width=True,
):
    st.cache_data.clear()
    st.success("캐시를 삭제했습니다.")
    st.rerun()
st.write(
    "입력 영상과 후보 영상의 자막을 TF-IDF 벡터로 변환하고, "
    "코사인 유사도를 이용해 관련 영상을 추천합니다."
)

with st.expander("📐 추천 모델의 계산 원리"):
    st.markdown(
        """
### 1. TF-IDF

단어 \(t\), 문서 \(d\)에 대해 다음과 같이 계산합니다.

\[
TFIDF(t,d)=TF(t,d)\\times IDF(t)
\]

### 2. 코사인 유사도

\[
CosineSimilarity(A,B)
=
\\frac{A\\cdot B}
{\\lVert A\\rVert\\lVert B\\rVert}
\]

### 3. 텍스트 유사도

\[
TextSimilarity
=
0.3\\times TitleSimilarity
+
0.7\\times ContentSimilarity
\]

### 4. 최종 추천 점수

\[
FinalScore
=
0.9\\times TextSimilarity
+
0.1\\times Popularity
\]

추천 결과의 관련성을 우선하기 위해 사용자가 가중치를 변경하지
못하도록 고정했습니다.
        """
    )

with st.sidebar:
    st.header("분석 설정")

    summary_sentence_count = st.slider(
        "요약 문장 수",
        min_value=3,
        max_value=10,
        value=5,
    )

    candidate_count = st.slider(
        "분석할 후보 영상 수",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
    )

    recommendation_count = st.slider(
        "표시할 추천 결과 수",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.info(
        "추천 가중치는 관련성 향상을 위해 고정되어 있습니다.\n\n"
        "- 텍스트 유사도: 90%\n"
        "- 인기도: 10%\n"
        "- 최소 유사도: 0.15"
    )

api_key = get_api_key()

if not api_key:
    st.warning(
        "Streamlit Secrets에 YOUTUBE_API_KEY를 등록하세요."
    )

youtube_url = st.text_input(
    "분석할 YouTube 영상 URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

button_column1, button_column2 = st.columns(2)

with button_column1:
    diagnostic_button = st.button(
        "🔍 자막 연결 상태 진단",
        use_container_width=True,
    )

with button_column2:
    analyze_button = st.button(
        "📊 TF-IDF 분석 및 추천 실행",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 자막 연결 상태 진단
# =========================================================

if diagnostic_button:
    if not youtube_url.strip():
        st.error("먼저 YouTube 영상 URL을 입력하세요.")

    else:
        diagnostic_video_id = extract_video_id(youtube_url)

        if not diagnostic_video_id:
            st.error("올바른 YouTube 영상 URL이 아닙니다.")

        else:
            with st.spinner(
                "YouTube 자막 연결 상태를 확인하는 중입니다..."
            ):
                diagnostic_result = get_transcript(
                    diagnostic_video_id
                )

            if diagnostic_result["success"]:
                st.success("자막을 정상적으로 가져왔습니다.")

                information_column1, information_column2, information_column3 = (
                    st.columns(3)
                )

                with information_column1:
                    st.metric(
                        "자막 언어",
                        diagnostic_result.get(
                            "language",
                            "unknown",
                        ),
                    )

                with information_column2:
                    st.metric(
                        "자막 유형",
                        (
                            "자동 생성"
                            if diagnostic_result.get(
                                "generated",
                                False,
                            )
                            else "수동 등록"
                        ),
                    )

                with information_column3:
                    st.metric(
                        "번역 여부",
                        (
                            "번역 자막"
                            if diagnostic_result.get(
                                "translated",
                                False,
                            )
                            else "원본 자막"
                        ),
                    )

                transcript_text = diagnostic_result.get(
                    "text",
                    "",
                )

                st.write(
                    f"가져온 자막 길이: "
                    f"{len(transcript_text):,}자"
                )

                st.text_area(
                    "가져온 자막 미리보기",
                    transcript_text[:2000],
                    height=250,
                )

            else:
                st.error("자막을 가져오지 못했습니다.")

                error_type = diagnostic_result.get(
                    "error_type",
                    "UnknownError",
                )

                error_message = diagnostic_result.get(
                    "error_message",
                    "정확한 오류 정보를 확인하지 못했습니다.",
                )

                st.write(f"오류 유형: `{error_type}`")
                st.write(f"오류 설명: {error_message}")

                if error_type in {
                    "IpBlocked",
                    "RequestBlocked",
                }:
                    st.warning(
                        "영상에 자막이 있어도 YouTube가 "
                        "Streamlit Cloud 서버의 요청을 차단한 "
                        "상태일 수 있습니다."
                    )

                elif error_type == "TranscriptsDisabled":
                    st.warning(
                        "영상 소유자가 외부 자막 접근을 "
                        "비활성화했을 가능성이 있습니다."
                    )

                elif error_type == "AgeRestricted":
                    st.warning(
                        "연령 제한 영상은 로그인하지 않은 "
                        "서버에서 자막을 가져오기 어렵습니다."
                    )

                else:
                    st.info(
                        "다른 공개 영상으로 시험하거나 캐시를 "
                        "삭제한 뒤 다시 시도해 보세요."
                    )

if analyze_button:
    if not youtube_url.strip():
        st.error("YouTube 영상 URL을 입력하세요.")
        st.stop()

    if not api_key:
        st.error("YouTube Data API 키를 먼저 설정하세요.")
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("올바른 YouTube 영상 URL이 아닙니다.")
        st.stop()

    try:
        with st.spinner("입력 영상 정보와 자막을 가져오는 중입니다..."):
            source_details = get_video_details(
                video_id,
                api_key,
            )

            source_transcript = get_transcript(video_id)

        if not source_details:
            st.error("입력 영상 정보를 가져오지 못했습니다.")
            st.stop()

        if not source_transcript["success"]:
            st.error(
                "입력 영상의 자막을 가져오지 못했습니다. "
                "자막이 제공되는 영상을 사용하세요."
            )
            st.stop()

        source_content = source_transcript["text"]

        with st.spinner("입력 자막을 TF-IDF로 분석하는 중입니다..."):
            summary_result = summarize_with_tfidf(
                source_content,
                sentence_count=summary_sentence_count,
            )

        search_keywords = summary_result["keywords"][:5]

        search_query = " ".join(
            [source_details["title"]] + search_keywords
        )[:250]

        with st.spinner("관련 영상 후보를 검색하는 중입니다..."):
            candidates = search_candidate_videos(
                search_query,
                api_key,
                candidate_count,
            )

        candidates = [
            candidate
            for candidate in candidates
            if candidate["video_id"] != video_id
        ]

        if not candidates:
            st.error("후보 영상을 찾지 못했습니다.")
            st.stop()

        progress_bar = st.progress(0)
        progress_message = st.empty()

        for index, candidate in enumerate(candidates):
            progress_message.write(
                "후보 영상 자막 분석 중: "
                f"{index + 1}/{len(candidates)}"
            )

            transcript_result = get_transcript(
                candidate["video_id"]
            )

            candidate["transcript_available"] = (
                transcript_result["success"]
            )

            candidate["transcript_language"] = (
                transcript_result["language"]
            )

            if transcript_result["success"]:
                candidate["comparison_content"] = (
                    transcript_result["text"]
                )
            else:
                candidate["comparison_content"] = " ".join(
                    [
                        candidate["title"],
                        candidate["title"],
                        candidate["description"],
                    ]
                )

            progress_bar.progress(
                (index + 1) / len(candidates)
            )

        progress_bar.empty()
        progress_message.empty()

        with st.spinner(
            "TF-IDF 벡터와 코사인 유사도를 계산하는 중입니다..."
        ):
            recommendation_result = recommend_videos(
                source_title=source_details["title"],
                source_content=source_content,
                candidates=candidates,
            )

        results_df = recommendation_result["results"]
        orange_df = recommendation_result["orange_features"]

        st.success("분석이 완료되었습니다.")

        if (
            not results_df.empty
            and results_df["fallback_used"].iloc[0]
        ):
            st.warning(
                "최소 유사도 0.15를 통과한 영상이 없어 "
                "후보 중 유사도가 높은 결과를 표시합니다."
            )

        # 입력 영상
        st.subheader("1. 입력 영상")

        video_column, info_column = st.columns([1, 2])

        with video_column:
            st.video(source_details["url"])

        with info_column:
            st.markdown(f"### {source_details['title']}")
            st.write(
                f"채널: {source_details['channel_title']}"
            )
            st.write(
                f"자막 언어: {source_transcript['language']}"
            )
            st.write(
                f"조회수: {source_details['views']:,}"
            )
            st.write(
                f"좋아요: {source_details['likes']:,}"
            )

        # 요약 결과
        st.subheader("2. TF-IDF 핵심 내용 요약")
        st.write(summary_result["summary"])

        st.subheader("3. TF-IDF 핵심 키워드")

        st.dataframe(
            summary_result["keyword_scores"],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("문장별 TF-IDF 중요도 확인"):
            st.dataframe(
                summary_result["sentence_scores"],
                use_container_width=True,
                hide_index=True,
            )

        # 추천 결과
        st.subheader("4. 코사인 유사도 기반 추천 결과")

        shown_df = results_df.head(
            recommendation_count
        ).copy()

        for index, row in shown_df.iterrows():
            st.markdown("---")

            image_column, result_column = st.columns([1, 3])

            with image_column:
                if row["thumbnail"]:
                    st.image(
                        row["thumbnail"],
                        use_container_width=True,
                    )

            with result_column:
                st.markdown(
                    f"### {index + 1}. "
                    f"[{row['title']}]({row['url']})"
                )

                st.write(f"채널: {row['channel_title']}")

                transcript_status = (
                    "자막 사용"
                    if row["transcript_available"]
                    else "자막 없음: 제목·설명 사용"
                )

                st.write(f"비교 데이터: {transcript_status}")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "제목 유사도",
                    f"{row['title_similarity']:.3f}",
                )

                metric2.metric(
                    "자막·본문 유사도",
                    f"{row['content_similarity']:.3f}",
                )

                metric3.metric(
                    "결합 유사도",
                    f"{row['similarity']:.3f}",
                )

                st.write(
                    f"최종 추천 점수: {row['final_score']:.3f} | "
                    f"조회수: {int(row['views']):,} | "
                    f"좋아요: {int(row['likes']):,}"
                )

        # TF-IDF 계산 상세
        st.subheader("5. TF-IDF 계산 과정")

        if not shown_df.empty:
            selected_row = shown_df.iloc[0]

            selected_candidate = next(
                candidate
                for candidate in candidates
                if candidate["video_id"]
                == selected_row["video_id"]
            )

            tfidf_detail_df = create_tfidf_explanation_table(
                source_content,
                selected_candidate["comparison_content"],
                maximum_terms=30,
            )

            st.write(
                "아래 표는 입력 영상과 추천 1위 영상에서 "
                "중요도가 높은 단어의 TF, IDF, TF-IDF 값입니다."
            )

            st.dataframe(
                tfidf_detail_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                f"""
**추천 1위 영상의 코사인 유사도 계산 결과**

- 제목 벡터 코사인 유사도:
  `{selected_row['title_similarity']:.6f}`
- 자막·본문 벡터 코사인 유사도:
  `{selected_row['content_similarity']:.6f}`
- 결합 유사도:

\[
0.3\\times
{selected_row['title_similarity']:.6f}
+
0.7\\times
{selected_row['content_similarity']:.6f}
=
{selected_row['similarity']:.6f}
\]
                """
            )

        # 그래프
        st.subheader("6. 추천 영상 유사도 비교")

        chart_df = shown_df[
            [
                "title",
                "title_similarity",
                "content_similarity",
                "similarity",
                "final_score",
            ]
        ].set_index("title")

        st.bar_chart(chart_df)

        # 다운로드
        st.subheader("7. 데이터 다운로드")

        recommendation_export = results_df.drop(
            columns=["comparison_content"],
            errors="ignore",
        )

        column1, column2, column3 = st.columns(3)

        with column1:
            st.download_button(
                "추천 결과 CSV",
                data=dataframe_to_csv_bytes(
                    recommendation_export
                ),
                file_name="recommendation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with column2:
            st.download_button(
                "Orange3 TF-IDF CSV",
                data=dataframe_to_csv_bytes(orange_df),
                file_name="orange_tfidf_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with column3:
            st.download_button(
                "TF-IDF 계산표 CSV",
                data=dataframe_to_csv_bytes(
                    tfidf_detail_df
                ),
                file_name="tfidf_calculation.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"분석 중 오류가 발생했습니다: {error}")

        st.info(
            "자막 제공 여부, API 할당량, 영상 공개 상태를 "
            "확인한 뒤 다시 시도하세요."
        )
