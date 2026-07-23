import os
import re
import traceback
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
# 1. 페이지 및 추천 모델 설정
# =========================================================

st.set_page_config(
    page_title="YouTube TF-IDF 영상 추천",
    page_icon="🎬",
    layout="wide",
)

TEXT_WEIGHT = 0.90
POPULARITY_WEIGHT = 0.10

TITLE_WEIGHT = 0.40
CONTENT_WEIGHT = 0.60

MINIMUM_SIMILARITY = 0.10
MAX_TFIDF_FEATURES = 1000
MAX_TRANSCRIPT_CHARACTERS = 30000


# =========================================================
# 2. 불용어
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
    "하는데", "보면", "한번", "아니", "그러면", "그냥",
    "지금", "여기", "이렇게", "저렇게", "뭐", "또", "아주",
    "많이", "있고", "없고", "위해", "관련", "경우", "이번",
}


# =========================================================
# 3. 텍스트 전처리
# =========================================================

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*(음악|박수|웃음)[^\)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def limit_text(text, maximum_characters=MAX_TRANSCRIPT_CHARACTERS):
    text = clean_text(text)
    return text[:maximum_characters]


def tokenize_mixed_text(text):
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
        if 15 <= len(sentence.strip()) <= 600
    ]

    if len(sentences) <= 1 and len(text) > 400:
        chunk_size = 250

        sentences = [
            text[index:index + chunk_size].strip()
            for index in range(0, len(text), chunk_size)
            if len(text[index:index + chunk_size].strip()) >= 15
        ]

    return sentences


# =========================================================
# 4. 업로드 자막 파일 처리
# =========================================================

def decode_uploaded_file(uploaded_file):
    raw_data = uploaded_file.getvalue()

    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("자막 파일의 문자 인코딩을 읽지 못했습니다.")


def clean_subtitle_file(text):
    if not text:
        return ""

    text = re.sub(
        r"^\s*WEBVTT.*?$",
        " ",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*\d+\s*$",
        " ",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}"
        r"\s*-->\s*"
        r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}.*$",
        " ",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"\d{1,2}:\d{2}[,.]\d{3}"
        r"\s*-->\s*"
        r"\d{1,2}:\d{2}[,.]\d{3}.*$",
        " ",
        text,
        flags=re.MULTILINE,
    )

    lines = []
    previous_line = ""

    for line in text.splitlines():
        line = clean_text(line)

        if not line or line == previous_line:
            continue

        lines.append(line)
        previous_line = line

    return limit_text(" ".join(lines))


def read_uploaded_subtitle(uploaded_file):
    raw_text = decode_uploaded_file(uploaded_file)
    subtitle_text = clean_subtitle_file(raw_text)

    if len(tokenize_mixed_text(subtitle_text)) < 5:
        raise ValueError("업로드한 파일의 자막 텍스트가 부족합니다.")

    return subtitle_text


# =========================================================
# 5. YouTube URL 및 API
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

        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) >= 2 and path_parts[0] in {
            "shorts",
            "embed",
            "live",
        }:
            candidate = path_parts[1]
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

    days = (datetime.now(timezone.utc) - published_date).days

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
def search_candidate_videos(query, api_key, max_results=20):
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
# 6. 입력 영상 자막 수집
# 입력 영상 한 개만 요청합니다.
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_transcript(video_id):
    def create_result(
        success=False,
        text="",
        language="none",
        generated=False,
        translated=False,
        error_type="",
        error_message="",
        available_transcripts=None,
    ):
        return {
            "success": success,
            "text": text,
            "language": language,
            "generated": generated,
            "translated": translated,
            "error_type": error_type,
            "error_message": error_message,
            "available_transcripts": available_transcripts or [],
        }

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        transcripts = list(transcript_list)

    except Exception as error:
        return create_result(
            error_type=type(error).__name__,
            error_message=str(error) or repr(error),
        )

    available_transcripts = [
        {
            "language": getattr(
                transcript,
                "language",
                "unknown",
            ),
            "language_code": getattr(
                transcript,
                "language_code",
                "unknown",
            ),
            "is_generated": getattr(
                transcript,
                "is_generated",
                False,
            ),
            "is_translatable": getattr(
                transcript,
                "is_translatable",
                False,
            ),
        }
        for transcript in transcripts
    ]

    if not transcripts:
        return create_result(
            error_type="EmptyTranscriptList",
            error_message="자막 목록이 비어 있습니다.",
            available_transcripts=available_transcripts,
        )

    def transcript_priority(transcript):
        language_code = getattr(
            transcript,
            "language_code",
            "",
        ).lower()

        generated = getattr(
            transcript,
            "is_generated",
            False,
        )

        if language_code.startswith("ko") and not generated:
            return 0

        if language_code.startswith("ko") and generated:
            return 1

        if language_code.startswith("en") and not generated:
            return 2

        if language_code.startswith("en") and generated:
            return 3

        if not generated:
            return 4

        return 5

    transcripts.sort(key=transcript_priority)
    errors = []

    for transcript in transcripts:
        language_code = getattr(
            transcript,
            "language_code",
            "unknown",
        )

        try:
            fetched = transcript.fetch()
            text_parts = []

            for snippet in fetched:
                snippet_text = getattr(snippet, "text", "")

                if snippet_text:
                    text_parts.append(
                        str(snippet_text).replace("\n", " ")
                    )

            text = clean_text(" ".join(text_parts))

            if len(tokenize_mixed_text(text)) >= 5:
                return create_result(
                    success=True,
                    text=limit_text(text),
                    language=language_code,
                    generated=getattr(
                        transcript,
                        "is_generated",
                        False,
                    ),
                    translated=False,
                    available_transcripts=available_transcripts,
                )

            errors.append(
                f"{language_code}: 자막 텍스트가 비어 있음"
            )

        except Exception as error:
            errors.append(
                f"{language_code}: "
                f"{type(error).__name__}: {error}"
            )

    for transcript in transcripts:
        if not getattr(transcript, "is_translatable", False):
            continue

        source_language = getattr(
            transcript,
            "language_code",
            "unknown",
        )

        for target_language in ["ko", "en"]:
            try:
                translated = transcript.translate(target_language)
                fetched = translated.fetch()

                text = clean_text(
                    " ".join(
                        str(getattr(snippet, "text", "")).replace(
                            "\n",
                            " ",
                        )
                        for snippet in fetched
                        if getattr(snippet, "text", "")
                    )
                )

                if len(tokenize_mixed_text(text)) >= 5:
                    return create_result(
                        success=True,
                        text=limit_text(text),
                        language=target_language,
                        generated=getattr(
                            transcript,
                            "is_generated",
                            False,
                        ),
                        translated=True,
                        available_transcripts=available_transcripts,
                    )

            except Exception as error:
                errors.append(
                    f"{source_language} → {target_language}: "
                    f"{type(error).__name__}: {error}"
                )

    return create_result(
        error_type="TranscriptFetchFailed",
        error_message="\n".join(errors),
        available_transcripts=available_transcripts,
    )


# =========================================================
# 7. TF-IDF 요약
# =========================================================

def summarize_with_tfidf(
    text,
    sentence_count=5,
    keyword_count=20,
):
    sentences = split_sentences(text)

    if not sentences:
        raise ValueError("TF-IDF로 분석할 문장이 없습니다.")

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
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

    sentence_df = pd.DataFrame(
        [
            {
                "중요도 순위": rank + 1,
                "원래 순서": int(index + 1),
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
            "단어 또는 구": keywords,
            "TF-IDF 점수": [
                round(float(total_term_scores[index]), 6)
                for index in keyword_indices
            ],
        }
    )

    return {
        "summary": summary,
        "keywords": keywords,
        "sentence_scores": sentence_df,
        "keyword_scores": keyword_df,
    }


# =========================================================
# 8. 추천 모델
# =========================================================

def normalize_log_values(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    logged = np.log1p(values).reshape(-1, 1)

    if len(values) == 1 or np.allclose(logged, logged[0]):
        return np.zeros(len(values))

    return MinMaxScaler().fit_transform(logged).ravel()


def calculate_recommendations(
    source_title,
    source_content,
    candidates,
):
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()

    candidate_titles = [
        candidate["title"]
        for candidate in candidates
    ]

    # 후보 영상은 자막 요청을 하지 않고 제목과 설명을 사용합니다.
    candidate_contents = [
        " ".join(
            [
                candidate["title"],
                candidate["title"],
                candidate["description"],
            ]
        )
        for candidate in candidates
    ]

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

    content_vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=MAX_TFIDF_FEATURES,
    )

    # 입력 영상은 전체 자막보다 요약·키워드를 사용하는 편이
    # 후보 제목·설명과 길이가 비슷해 비교가 더 안정적입니다.
    content_matrix = content_vectorizer.fit_transform(
        [source_content] + candidate_contents
    )

    content_similarities = cosine_similarity(
        content_matrix[0:1],
        content_matrix[1:],
    ).ravel()

    text_similarities = (
        TITLE_WEIGHT * title_similarities
        + CONTENT_WEIGHT * content_similarities
    )

    views = [
        candidate["daily_views"]
        for candidate in candidates
    ]

    likes = [
        candidate["likes"]
        for candidate in candidates
    ]

    normalized_views = normalize_log_values(views)
    normalized_likes = normalize_log_values(likes)

    popularity_scores = (
        0.6 * normalized_views
        + 0.4 * normalized_likes
    )

    final_scores = (
        TEXT_WEIGHT * text_similarities
        + POPULARITY_WEIGHT * popularity_scores
    )

    rows = []

    for index, candidate in enumerate(candidates):
        row = candidate.copy()
        row["title_similarity"] = float(
            title_similarities[index]
        )
        row["content_similarity"] = float(
            content_similarities[index]
        )
        row["similarity"] = float(
            text_similarities[index]
        )
        row["popularity_score"] = float(
            popularity_scores[index]
        )
        row["final_score"] = float(final_scores[index])
        row["comparison_source"] = "제목·설명"

        rows.append(row)

    results = pd.DataFrame(rows)

    passed_results = results[
        results["similarity"] >= MINIMUM_SIMILARITY
    ].copy()

    fallback_used = False

    if passed_results.empty:
        fallback_used = True
        passed_results = results.nlargest(
            min(5, len(results)),
            "similarity",
        ).copy()

    passed_results["fallback_used"] = fallback_used

    # 관련성 자체를 가장 먼저 고려합니다.
    passed_results = passed_results.sort_values(
        by=["similarity", "final_score"],
        ascending=False,
    ).reset_index(drop=True)

    feature_names = content_vectorizer.get_feature_names_out()

    feature_columns = [
        f"tfidf_{index:04d}_{term.replace(' ', '_')}"
        for index, term in enumerate(feature_names)
    ]

    feature_df = pd.DataFrame(
        content_matrix[1:].toarray(),
        columns=feature_columns,
    )

    metadata_df = results[
        [
            "video_id",
            "title",
            "channel_title",
            "url",
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
            metadata_df.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return passed_results, orange_df


def create_tfidf_detail_table(
    source_text,
    comparison_text,
    maximum_terms=30,
):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        use_idf=True,
        smooth_idf=True,
        norm=None,
        max_features=MAX_TFIDF_FEATURES,
    )

    matrix = vectorizer.fit_transform(
        [source_text, comparison_text]
    )

    terms = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_

    source_values = matrix[0].toarray().ravel()
    comparison_values = matrix[1].toarray().ravel()

    dataframe = pd.DataFrame(
        {
            "단어": terms,
            "IDF": idf_values,
            "입력 영상 TF-IDF": source_values,
            "추천 영상 TF-IDF": comparison_values,
            "TF-IDF 합계": source_values + comparison_values,
        }
    )

    return dataframe.sort_values(
        "TF-IDF 합계",
        ascending=False,
    ).head(maximum_terms)


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# =========================================================
# 9. 화면 구성
# =========================================================

st.title("🎬 YouTube TF-IDF 자막 분석 및 영상 추천")

st.write(
    "입력 영상은 자막을 분석하고, 후보 영상은 제목과 설명을 "
    "TF-IDF 벡터로 변환하여 코사인 유사도를 계산합니다."
)

if st.sidebar.button(
    "🗑️ 캐시 삭제",
    use_container_width=True,
):
    st.cache_data.clear()
    st.sidebar.success("캐시를 삭제했습니다.")

with st.sidebar:
    st.header("분석 설정")

    summary_sentence_count = st.slider(
        "요약 문장 수",
        min_value=3,
        max_value=10,
        value=5,
    )

    candidate_count = st.slider(
        "검색할 후보 영상 수",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    recommendation_count = st.slider(
        "표시할 추천 영상 수",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.info(
        "추천 모델 고정값\n\n"
        "- 텍스트 관련성: 90%\n"
        "- 인기도: 10%\n"
        "- 제목 유사도: 40%\n"
        "- 내용 유사도: 60%\n"
        "- 최소 유사도: 0.10"
    )

with st.expander("📐 TF-IDF 및 추천 점수 계산 원리"):
    st.markdown(
        r"""
### TF-IDF

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

### 코사인 유사도

\[
CosineSimilarity(A,B)
=
\frac{A\cdot B}
{\|A\|\|B\|}
\]

### 텍스트 유사도

\[
TextSimilarity
=
0.4\times TitleSimilarity
+
0.6\times ContentSimilarity
\]

### 최종 점수

\[
FinalScore
=
0.9\times TextSimilarity
+
0.1\times Popularity
\]
        """
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

st.subheader("자막 대체 입력")

uploaded_subtitle = st.file_uploader(
    "자동 자막 수집 실패 시 TXT, SRT 또는 VTT 파일 업로드",
    type=["txt", "srt", "vtt"],
)

manual_transcript = st.text_area(
    "또는 자막을 직접 붙여 넣으세요.",
    height=120,
)

button_column1, button_column2 = st.columns(2)

with button_column1:
    diagnostic_button = st.button(
        "🔍 입력 영상 자막 진단",
        use_container_width=True,
    )

with button_column2:
    analyze_button = st.button(
        "📊 TF-IDF 분석 및 추천",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 10. 자막 진단
# =========================================================

if diagnostic_button:
    diagnostic_video_id = extract_video_id(youtube_url)

    if not diagnostic_video_id:
        st.error("올바른 YouTube URL을 입력하세요.")

    else:
        with st.spinner("입력 영상 자막을 한 번 확인하고 있습니다..."):
            diagnostic_result = get_transcript(
                diagnostic_video_id
            )

        if diagnostic_result["success"]:
            st.success("자막을 정상적으로 가져왔습니다.")

            column1, column2, column3 = st.columns(3)

            column1.metric(
                "언어",
                diagnostic_result["language"],
            )

            column2.metric(
                "자막 유형",
                (
                    "자동 생성"
                    if diagnostic_result["generated"]
                    else "수동 등록"
                ),
            )

            column3.metric(
                "번역 여부",
                (
                    "번역"
                    if diagnostic_result["translated"]
                    else "원본"
                ),
            )

            st.text_area(
                "자막 미리보기",
                diagnostic_result["text"][:2000],
                height=250,
            )

            if diagnostic_result["available_transcripts"]:
                st.dataframe(
                    pd.DataFrame(
                        diagnostic_result[
                            "available_transcripts"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.error(
                "YouTube 자막 자동 수집에 실패했습니다. "
                "영상에 자막이 없다는 뜻은 아닐 수 있습니다."
            )

            st.write(
                "오류 유형: "
                f"`{diagnostic_result['error_type']}`"
            )

            st.code(
                diagnostic_result["error_message"]
                or "오류 메시지가 없습니다."
            )

            if diagnostic_result["available_transcripts"]:
                st.write("발견된 자막 목록")

                st.dataframe(
                    pd.DataFrame(
                        diagnostic_result[
                            "available_transcripts"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            if diagnostic_result["error_type"] in {
                "IpBlocked",
                "RequestBlocked",
                "TooManyRequests",
            }:
                st.warning(
                    "YouTube가 Streamlit Cloud 서버의 요청을 "
                    "제한한 상태입니다. 자막을 직접 입력하거나 "
                    "파일을 업로드하세요."
                )


# =========================================================
# 11. 분석 및 추천 실행
# =========================================================

if analyze_button:
    if not api_key:
        st.error("YouTube Data API 키를 먼저 설정하세요.")
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("올바른 YouTube URL을 입력하세요.")
        st.stop()

    try:
        with st.spinner("입력 영상 정보를 가져오는 중입니다..."):
            source_details = get_video_details(
                video_id,
                api_key,
            )

        if not source_details:
            st.error("영상 정보를 가져오지 못했습니다.")
            st.stop()

        source_content = ""
        source_data_label = ""
        transcript_result = None

        # 사용자가 제공한 자막을 우선합니다.
        if manual_transcript.strip():
            source_content = limit_text(manual_transcript)
            source_data_label = "사용자가 직접 입력한 자막"

        elif uploaded_subtitle is not None:
            source_content = read_uploaded_subtitle(
                uploaded_subtitle
            )

            source_data_label = (
                f"업로드 자막: {uploaded_subtitle.name}"
            )

        else:
            with st.spinner(
                "입력 영상의 YouTube 자막을 한 번 요청하는 중입니다..."
            ):
                transcript_result = get_transcript(video_id)

            if transcript_result["success"]:
                source_content = transcript_result["text"]

                if transcript_result["translated"]:
                    source_data_label = "YouTube 번역 자막"

                elif transcript_result["generated"]:
                    source_data_label = "YouTube 자동 생성 자막"

                else:
                    source_data_label = "YouTube 수동 등록 자막"

            else:
                fallback_text = " ".join(
                    [
                        source_details["title"],
                        source_details["title"],
                        source_details["description"],
                    ]
                )

                if len(tokenize_mixed_text(fallback_text)) < 5:
                    st.error(
                        "자막 자동 수집에 실패했고 영상 설명도 "
                        "부족합니다. 자막을 직접 입력하거나 "
                        "파일을 업로드하세요."
                    )

                    st.write(
                        "자막 오류: "
                        f"`{transcript_result['error_type']}`"
                    )

                    st.stop()

                source_content = fallback_text
                source_data_label = "자막 수집 실패: 제목·설명 사용"

                st.warning(
                    "YouTube 자막 자동 수집에 실패하여 "
                    "제목과 설명으로 분석합니다."
                )

                st.write(
                    "자막 오류 유형: "
                    f"`{transcript_result['error_type']}`"
                )

        with st.spinner("입력 텍스트를 TF-IDF로 분석 중입니다..."):
            summary_result = summarize_with_tfidf(
                source_content,
                sentence_count=summary_sentence_count,
            )

        search_keywords = summary_result["keywords"][:5]

        search_query = " ".join(
            [source_details["title"]] + search_keywords
        )[:250]

        with st.spinner("관련 영상 후보를 검색 중입니다..."):
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
            st.error("추천 후보 영상을 찾지 못했습니다.")
            st.stop()

        # 후보 자막은 요청하지 않습니다.
        recommendation_source_text = " ".join(
            [
                source_details["title"],
                source_details["title"],
                summary_result["summary"],
                " ".join(summary_result["keywords"]),
            ]
        )

        with st.spinner(
            "TF-IDF 벡터와 코사인 유사도를 계산 중입니다..."
        ):
            results_df, orange_df = calculate_recommendations(
                source_title=source_details["title"],
                source_content=recommendation_source_text,
                candidates=candidates,
            )

        if results_df.empty:
            st.error("추천 결과를 만들지 못했습니다.")
            st.stop()

        st.success("분석이 완료되었습니다.")

        if results_df["fallback_used"].iloc[0]:
            st.warning(
                "최소 유사도 기준을 통과한 영상이 없어 "
                "후보 중 상대적으로 유사한 영상을 표시합니다."
            )

        st.subheader("1. 입력 영상")

        video_column, information_column = st.columns([1, 2])

        with video_column:
            st.video(source_details["url"])

        with information_column:
            st.markdown(f"### {source_details['title']}")
            st.write(
                f"채널: {source_details['channel_title']}"
            )
            st.write(f"분석 데이터: {source_data_label}")
            st.write(
                f"조회수: {source_details['views']:,}"
            )
            st.write(
                f"좋아요: {source_details['likes']:,}"
            )

        st.subheader("2. TF-IDF 핵심 내용 요약")
        st.write(summary_result["summary"])

        st.subheader("3. TF-IDF 핵심 키워드")

        st.dataframe(
            summary_result["keyword_scores"],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("문장별 TF-IDF 중요도"):
            st.dataframe(
                summary_result["sentence_scores"],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("4. 관련 영상 추천")

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
                st.write("비교 데이터: 후보 영상 제목·설명")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "제목 유사도",
                    f"{row['title_similarity']:.3f}",
                )

                metric2.metric(
                    "내용 유사도",
                    f"{row['content_similarity']:.3f}",
                )

                metric3.metric(
                    "결합 유사도",
                    f"{row['similarity']:.3f}",
                )

                st.write(
                    f"최종 점수: {row['final_score']:.3f} | "
                    f"조회수: {int(row['views']):,} | "
                    f"좋아요: {int(row['likes']):,}"
                )

        st.subheader("5. TF-IDF 계산 상세")

        first_result = shown_df.iloc[0]

        first_candidate = next(
            candidate
            for candidate in candidates
            if candidate["video_id"]
            == first_result["video_id"]
        )

        comparison_text = " ".join(
            [
                first_candidate["title"],
                first_candidate["title"],
                first_candidate["description"],
            ]
        )

        detail_df = create_tfidf_detail_table(
            recommendation_source_text,
            comparison_text,
        )

        st.write(
            "입력 영상과 추천 1위 영상에서 중요도가 높은 "
            "단어의 IDF와 TF-IDF 값입니다."
        )

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("6. 유사도 비교 그래프")

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

        st.subheader("7. CSV 다운로드")

        download_column1, download_column2, download_column3 = (
            st.columns(3)
        )

        with download_column1:
            st.download_button(
                "추천 결과 CSV",
                data=dataframe_to_csv_bytes(results_df),
                file_name="recommendation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with download_column2:
            st.download_button(
                "Orange3 TF-IDF CSV",
                data=dataframe_to_csv_bytes(orange_df),
                file_name="orange_tfidf_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with download_column3:
            st.download_button(
                "TF-IDF 상세 CSV",
                data=dataframe_to_csv_bytes(detail_df),
                file_name="tfidf_details.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as error:
        st.error("분석 중 오류가 발생했습니다.")
        st.write(f"오류 유형: `{type(error).__name__}`")
        st.write(f"오류 내용: {error}")

        with st.expander("개발자용 상세 오류"):
            st.code(traceback.format_exc())
