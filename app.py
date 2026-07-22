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
# Streamlit 기본 설정
# =========================================================

st.set_page_config(
    page_title="YouTube 자막 요약 및 영상 추천",
    page_icon="🎬",
    layout="wide",
)


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
}


# =========================================================
# 텍스트 전처리 함수
# =========================================================

def clean_text(text):
    """자막과 영상 설명의 불필요한 문자를 제거합니다."""
    if not text:
        return ""

    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*(음악|박수|웃음)[^\)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text):
    """한국어와 영어 자막을 문장 단위로 분리합니다."""
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+|(?<=[다요죠])\s+(?=[가-힣A-Z0-9])",
        text,
    )

    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if 15 <= len(sentence) <= 500:
            cleaned_sentences.append(sentence)

    # 문장 부호가 없는 자동 자막에 대비해 일정 길이로 분할합니다.
    if len(cleaned_sentences) <= 1 and len(text) > 500:
        chunk_size = 250

        cleaned_sentences = [
            text[index:index + chunk_size].strip()
            for index in range(0, len(text), chunk_size)
            if len(text[index:index + chunk_size].strip()) >= 15
        ]

    return cleaned_sentences


def tokenize_mixed_text(text):
    """한국어 단어와 영어 단어를 함께 추출합니다."""
    text = clean_text(text).lower()

    tokens = re.findall(
        r"[가-힣]{2,}|[a-zA-Z][a-zA-Z0-9_-]{1,}",
        text,
    )

    filtered_tokens = []

    for token in tokens:
        if token in KOREAN_STOP_WORDS:
            continue

        if token in ENGLISH_STOP_WORDS:
            continue

        filtered_tokens.append(token)

    return filtered_tokens


# =========================================================
# YouTube 관련 함수
# =========================================================

def extract_video_id(url):
    """다양한 YouTube URL에서 영상 ID를 추출합니다."""
    if not url:
        return None

    url = url.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    hostname = hostname.replace("www.", "")

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

        if (
            len(path_parts) >= 2
            and path_parts[0] in {"shorts", "embed", "live"}
        ):
            candidate = path_parts[1]

            if len(candidate) == 11:
                return candidate

    match = re.search(
        r"(?:v=|youtu\.be/|shorts/|embed/|live/)"
        r"([A-Za-z0-9_-]{11})",
        url,
    )

    return match.group(1) if match else None


def get_youtube_client(api_key):
    """YouTube Data API 클라이언트를 생성합니다."""
    return build(
        "youtube",
        "v3",
        developerKey=api_key,
        cache_discovery=False,
    )


def parse_iso_datetime(value):
    """YouTube 날짜 문자열을 datetime으로 변환합니다."""
    if not value:
        return datetime.now(timezone.utc)

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def calculate_days_since_upload(published_at):
    """영상 업로드 이후 지난 일수를 계산합니다."""
    published_date = parse_iso_datetime(published_at)
    current_date = datetime.now(timezone.utc)
    days = (current_date - published_date).days

    return max(days, 1)


def get_thumbnail(snippet):
    """가장 적절한 썸네일 URL을 반환합니다."""
    thumbnails = snippet.get("thumbnails", {})

    return (
        thumbnails.get("high", {}).get("url")
        or thumbnails.get("medium", {}).get("url")
        or thumbnails.get("default", {}).get("url")
        or ""
    )


def get_api_key():
    """Streamlit Secrets 또는 환경 변수에서 API 키를 가져옵니다."""
    try:
        key = st.secrets["YOUTUBE_API_KEY"]

        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass

    return os.getenv("YOUTUBE_API_KEY", "")


@st.cache_data(ttl=3600, show_spinner=False)
def get_transcript(video_id):
    """한국어, 영어 순서로 영상 자막을 가져옵니다."""
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(
            video_id,
            languages=["ko", "en", "en-US", "en-GB"],
        )

        snippets = list(fetched)

        text = " ".join(
            snippet.text.replace("\n", " ")
            for snippet in snippets
        )

        return {
            "text": clean_text(text),
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
        except Exception as error:
            raise ValueError(
                "자막을 가져올 수 없습니다. 자막 제공 여부를 확인하세요."
            ) from error

        if not transcripts:
            raise ValueError(
                "이 영상에는 사용할 수 있는 자막이 없습니다."
            )

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
        snippets = list(fetched)

        text = " ".join(
            snippet.text.replace("\n", " ")
            for snippet in snippets
        )

        return {
            "text": clean_text(text),
            "language": selected.language_code,
        }


@st.cache_data(ttl=3600, show_spinner=False)
def get_video_details(video_id, api_key):
    """영상 제목, 설명, 조회수, 좋아요 등의 정보를 가져옵니다."""
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
    days_since_upload = calculate_days_since_upload(published_at)

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": published_at,
        "views": views,
        "likes": likes,
        "days_since_upload": days_since_upload,
        "daily_views": views / days_since_upload,
        "thumbnail": get_thumbnail(snippet),
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def search_candidate_videos(query, api_key, max_results=20):
    """검색어를 이용해 관련 영상 후보와 통계 정보를 가져옵니다."""
    youtube = get_youtube_client(api_key)

    search_response = (
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
        for item in search_response.get("items", [])
        if item.get("id", {}).get("videoId")
    ]

    if not video_ids:
        return []

    details_response = (
        youtube.videos()
        .list(
            part="snippet,statistics",
            id=",".join(video_ids),
        )
        .execute()
    )

    candidates = []

    for item in details_response.get("items", []):
        video_id = item.get("id", "")
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})

        views = int(statistics.get("viewCount", 0))
        likes = int(statistics.get("likeCount", 0))
        published_at = snippet.get("publishedAt", "")
        days_since_upload = calculate_days_since_upload(published_at)

        candidates.append(
            {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": published_at,
                "views": views,
                "likes": likes,
                "days_since_upload": days_since_upload,
                "daily_views": views / days_since_upload,
                "thumbnail": get_thumbnail(snippet),
                "url": (
                    "https://www.youtube.com/watch?v="
                    f"{video_id}"
                ),
            }
        )

    # 검색 API가 반환한 관련성 순서를 유지합니다.
    original_order = {
        video_id: index
        for index, video_id in enumerate(video_ids)
    }

    candidates.sort(
        key=lambda item: original_order.get(
            item["video_id"],
            len(video_ids),
        )
    )

    return candidates


# =========================================================
# TF-IDF 요약 함수
# =========================================================

def summarize_transcript(
    text,
    sentence_count=5,
    keyword_count=15,
):
    """TF-IDF 점수가 높은 문장을 선택해 추출 요약을 만듭니다."""
    sentences = split_sentences(text)

    if not sentences:
        raise ValueError("요약할 수 있는 자막 문장이 없습니다.")

    if len(sentences) <= sentence_count:
        return {
            "summary": " ".join(sentences),
            "keywords": [],
            "ranked_sentences": [
                {
                    "sentence": sentence,
                    "score": 1.0,
                    "original_order": index + 1,
                }
                for index, sentence in enumerate(sentences)
            ],
        }

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError as error:
        raise ValueError(
            "자막에서 TF-IDF 특징을 생성하지 못했습니다."
        ) from error

    sentence_scores = np.asarray(
        matrix.sum(axis=1)
    ).ravel()

    top_count = min(sentence_count, len(sentences))

    top_indices = np.argsort(
        sentence_scores
    )[::-1][:top_count]

    # 요약은 원래 영상에 등장한 문장 순서로 출력합니다.
    ordered_top_indices = sorted(top_indices.tolist())

    summary = " ".join(
        sentences[index]
        for index in ordered_top_indices
    )

    ranked_sentences = [
        {
            "sentence": sentences[index],
            "score": round(float(sentence_scores[index]), 4),
            "original_order": index + 1,
        }
        for index in top_indices
    ]

    feature_names = vectorizer.get_feature_names_out()
    term_scores = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    keyword_indices = np.argsort(term_scores)[::-1]
    keywords = []

    for index in keyword_indices:
        term = feature_names[index].strip()

        if len(term) < 2:
            continue

        keywords.append(term)

        if len(keywords) >= keyword_count:
            break

    return {
        "summary": summary,
        "keywords": keywords,
        "ranked_sentences": ranked_sentences,
    }


# =========================================================
# 추천 점수 계산 함수
# =========================================================

def normalize_log_values(values):
    """로그 변환 후 값을 0~1 범위로 정규화합니다."""
    values = np.asarray(values, dtype=float)
    logged_values = np.log1p(values).reshape(-1, 1)

    if (
        len(values) == 1
        or np.allclose(logged_values, logged_values[0])
    ):
        return np.zeros(len(values))

    scaler = MinMaxScaler()

    return scaler.fit_transform(logged_values).ravel()


def normalize_weights(
    similarity_weight,
    views_weight,
    likes_weight,
):
    """세 가중치의 합이 1이 되도록 정규화합니다."""
    weights = np.array(
        [
            similarity_weight,
            views_weight,
            likes_weight,
        ],
        dtype=float,
    )

    total = weights.sum()

    if total <= 0:
        return np.array([1.0, 0.0, 0.0])

    return weights / total


def build_orange_feature_dataframe(
    candidates,
    tfidf_matrix,
    vectorizer,
    similarities,
    normalized_daily_views,
    normalized_likes,
    final_scores,
):
    """Orange3 분석에 사용할 TF-IDF 특징 데이터프레임을 만듭니다."""
    feature_names = vectorizer.get_feature_names_out()
    dense_features = tfidf_matrix.toarray()

    safe_feature_names = [
        (
            f"tfidf_{index:03d}_"
            f"{name.replace(' ', '_')}"
        )
        for index, name in enumerate(feature_names)
    ]

    feature_df = pd.DataFrame(
        dense_features,
        columns=safe_feature_names,
    )

    metadata_df = pd.DataFrame(
        {
            "video_id": [
                candidate["video_id"]
                for candidate in candidates
            ],
            "title": [
                candidate["title"]
                for candidate in candidates
            ],
            "channel_title": [
                candidate["channel_title"]
                for candidate in candidates
            ],
            "url": [
                candidate["url"]
                for candidate in candidates
            ],
            "similarity": similarities,
            "views": [
                candidate["views"]
                for candidate in candidates
            ],
            "likes": [
                candidate["likes"]
                for candidate in candidates
            ],
            "daily_views": [
                candidate["daily_views"]
                for candidate in candidates
            ],
            "normalized_daily_views": normalized_daily_views,
            "normalized_likes": normalized_likes,
            "final_score": final_scores,
        }
    )

    return pd.concat(
        [
            metadata_df.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )


def recommend_videos(
    source_text,
    candidates,
    similarity_weight=0.7,
    views_weight=0.2,
    likes_weight=0.1,
    max_features=200,
):
    """내용 유사도와 영상 인기도를 결합해 추천 순위를 계산합니다."""
    if not candidates:
        return {
            "results": pd.DataFrame(),
            "orange_features": pd.DataFrame(),
        }

    candidate_texts = [
        f"{candidate['title']} {candidate['description']}"
        for candidate in candidates
    ]

    all_texts = [source_text] + candidate_texts

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_features,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError as error:
        raise ValueError(
            "입력 영상과 후보 영상에서 비교할 단어를 찾지 못했습니다."
        ) from error

    source_vector = tfidf_matrix[0:1]
    candidate_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(
        source_vector,
        candidate_vectors,
    ).ravel()

    daily_views = [
        candidate["daily_views"]
        for candidate in candidates
    ]

    likes = [
        candidate["likes"]
        for candidate in candidates
    ]

    normalized_daily_views = normalize_log_values(
        daily_views
    )

    normalized_likes = normalize_log_values(likes)

    weights = normalize_weights(
        similarity_weight,
        views_weight,
        likes_weight,
    )

    final_scores = (
        weights[0] * similarities
        + weights[1] * normalized_daily_views
        + weights[2] * normalized_likes
    )

    rows = []

    for index, candidate in enumerate(candidates):
        row = candidate.copy()

        row["similarity"] = round(
            float(similarities[index]),
            6,
        )

        row["normalized_daily_views"] = round(
            float(normalized_daily_views[index]),
            6,
        )

        row["normalized_likes"] = round(
            float(normalized_likes[index]),
            6,
        )

        row["final_score"] = round(
            float(final_scores[index]),
            6,
        )

        rows.append(row)

    results_df = pd.DataFrame(rows)

    results_df = results_df.sort_values(
        by=["final_score", "similarity"],
        ascending=False,
    ).reset_index(drop=True)

    orange_df = build_orange_feature_dataframe(
        candidates=candidates,
        tfidf_matrix=candidate_vectors,
        vectorizer=vectorizer,
        similarities=similarities,
        normalized_daily_views=normalized_daily_views,
        normalized_likes=normalized_likes,
        final_scores=final_scores,
    )

    return {
        "results": results_df,
        "orange_features": orange_df,
    }


def dataframe_to_csv_bytes(dataframe):
    """한글이 깨지지 않는 CSV 다운로드 데이터를 생성합니다."""
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# =========================================================
# Streamlit 화면
# =========================================================

st.title("🎬 YouTube 자막 요약 및 관련 영상 추천")

st.write(
    "YouTube 영상의 자막을 TF-IDF로 분석하여 핵심 문장을 "
    "추출하고, 코사인 유사도와 영상 인기도를 결합해 관련 "
    "영상을 추천합니다."
)

with st.sidebar:
    st.header("분석 설정")

    summary_sentence_count = st.slider(
        "요약 문장 수",
        min_value=3,
        max_value=15,
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
        max_value=15,
        value=5,
    )

    st.subheader("추천 점수 가중치")

    similarity_weight = st.slider(
        "텍스트 유사도",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
    )

    views_weight = st.slider(
        "일평균 조회수",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )

    likes_weight = st.slider(
        "좋아요",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
    )

    st.caption(
        "가중치의 합은 내부적으로 1이 되도록 자동 정규화됩니다."
    )

api_key = get_api_key()

if not api_key:
    st.warning(
        "YouTube Data API 키가 설정되지 않았습니다. "
        "Streamlit Cloud의 Secrets에 다음과 같이 등록하세요: "
        '`YOUTUBE_API_KEY = "API 키"`'
    )

youtube_url = st.text_input(
    "분석할 YouTube 영상 URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

analyze_button = st.button(
    "영상 분석하기",
    type="primary",
    use_container_width=True,
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
        st.error(
            "올바른 YouTube URL에서 영상 ID를 찾지 못했습니다."
        )
        st.stop()

    try:
        with st.spinner(
            "영상 정보와 자막을 가져오는 중입니다..."
        ):
            source_details = get_video_details(
                video_id,
                api_key,
            )

            transcript_result = get_transcript(video_id)

        if not source_details:
            st.error("영상 정보를 가져오지 못했습니다.")
            st.stop()

        transcript_text = transcript_result["text"]
        transcript_language = transcript_result["language"]

        with st.spinner(
            "TF-IDF로 핵심 문장을 추출하는 중입니다..."
        ):
            summary_result = summarize_transcript(
                transcript_text,
                sentence_count=summary_sentence_count,
            )

        source_title = source_details["title"]
        source_description = source_details["description"]

        # 핵심 키워드의 상위 8개를 후보 영상 검색에 사용합니다.
        query_terms = summary_result["keywords"][:8]
        search_query = " ".join(query_terms)

        if not search_query.strip():
            search_query = source_title

        with st.spinner(
            "관련 영상 후보를 검색하는 중입니다..."
        ):
            candidates = search_candidate_videos(
                search_query,
                api_key,
                candidate_count,
            )

        # 입력 영상 자체는 추천 목록에서 제외합니다.
        candidates = [
            video
            for video in candidates
            if video["video_id"] != video_id
        ]

        if not candidates:
            st.warning("추천할 후보 영상을 찾지 못했습니다.")
            st.stop()

        source_text = " ".join(
            [
                source_title,
                source_description,
                summary_result["summary"],
                transcript_text,
            ]
        )

        with st.spinner(
            "코사인 유사도와 추천 점수를 계산하는 중입니다..."
        ):
            recommendation_result = recommend_videos(
                source_text=source_text,
                candidates=candidates,
                similarity_weight=similarity_weight,
                views_weight=views_weight,
                likes_weight=likes_weight,
            )

        recommendations_df = recommendation_result["results"]
        orange_df = recommendation_result["orange_features"]

        st.success("분석이 완료되었습니다.")

        # 입력 영상 정보
        st.subheader("입력 영상")

        left_column, right_column = st.columns([1, 2])

        with left_column:
            st.video(
                f"https://www.youtube.com/watch?v={video_id}"
            )

        with right_column:
            st.markdown(f"### {source_title}")
            st.write(
                f"채널: {source_details['channel_title']}"
            )
            st.write(f"자막 언어: {transcript_language}")
            st.write(
                f"조회수: {source_details['views']:,}"
            )
            st.write(
                f"좋아요: {source_details['likes']:,}"
            )
            st.write(
                "일평균 조회수: "
                f"{source_details['daily_views']:,.1f}"
            )

        # 요약 출력
        st.subheader("TF-IDF 핵심 내용 요약")
        st.write(summary_result["summary"])

        # 키워드 출력
        st.subheader("핵심 키워드")

        if summary_result["keywords"]:
            st.write(
                " · ".join(summary_result["keywords"])
            )
        else:
            st.write("핵심 키워드를 추출하지 못했습니다.")

        # 핵심 문장 상세 정보
        with st.expander(
            "추출된 핵심 문장과 TF-IDF 점수 확인"
        ):
            sentence_df = pd.DataFrame(
                summary_result["ranked_sentences"]
            )

            st.dataframe(
                sentence_df,
                use_container_width=True,
                hide_index=True,
            )

        # 추천 영상 출력
        st.subheader("관련 영상 추천")

        shown_df = recommendations_df.head(
            recommendation_count
        ).copy()

        for index, row in shown_df.iterrows():
            st.markdown("---")

            thumbnail_column, information_column = st.columns(
                [1, 3]
            )

            with thumbnail_column:
                if row["thumbnail"]:
                    st.image(
                        row["thumbnail"],
                        use_container_width=True,
                    )

            with information_column:
                st.markdown(
                    f"### {index + 1}. "
                    f"[{row['title']}]({row['url']})"
                )

                st.write(
                    f"채널: {row['channel_title']}"
                )

                st.write(
                    f"코사인 유사도: {row['similarity']:.3f} | "
                    f"최종 추천 점수: {row['final_score']:.3f}"
                )

                st.write(
                    f"조회수: {int(row['views']):,} | "
                    f"좋아요: {int(row['likes']):,} | "
                    f"일평균 조회수: {row['daily_views']:,.1f}"
                )

        # 점수 비교 그래프
        st.subheader("추천 점수 비교")

        chart_df = shown_df[
            [
                "title",
                "similarity",
                "normalized_daily_views",
                "normalized_likes",
                "final_score",
            ]
        ].set_index("title")

        st.bar_chart(chart_df)

        # CSV 다운로드
        st.subheader("Orange3 분석 데이터")

        st.write(
            "추천 결과 CSV와 군집 분석용 TF-IDF 특징 CSV를 "
            "다운로드할 수 있습니다."
        )

        download_column1, download_column2 = st.columns(2)

        with download_column1:
            st.download_button(
                label="추천 결과 CSV 다운로드",
                data=dataframe_to_csv_bytes(
                    recommendations_df
                ),
                file_name="youtube_recommendations.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with download_column2:
            st.download_button(
                label="Orange3 TF-IDF CSV 다운로드",
                data=dataframe_to_csv_bytes(orange_df),
                file_name="orange_tfidf_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as error:
        st.error(
            f"분석 중 오류가 발생했습니다: {error}"
        )

        st.info(
            "영상의 자막 제공 여부, 공개 상태, 연령 제한 및 "
            "YouTube Data API 할당량을 확인하세요."
        )
