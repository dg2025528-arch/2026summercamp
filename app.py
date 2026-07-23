import os
import re
import traceback
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# 1. 페이지 및 모델 설정
# =========================================================

st.set_page_config(
    page_title="YouTube TF-IDF 분석 및 코사인 추천",
    page_icon="🎬",
    layout="wide",
)

MAX_TFIDF_FEATURES = 1500
MAX_TRANSCRIPT_CHARACTERS = 30000

# 이 값보다 낮은 영상은 추천하지 않습니다.
MINIMUM_COSINE_SIMILARITY = 0.08


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
    text = re.sub(
        r"\([^\)]*(음악|박수|웃음)[^\)]*\)",
        " ",
        text,
    )
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def limit_text(
    text,
    maximum_characters=MAX_TRANSCRIPT_CHARACTERS,
):
    return clean_text(text)[:maximum_characters]


def tokenize_mixed_text(text):
    """
    한국어·영어를 함께 처리하는 TF-IDF 토크나이저입니다.
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
        if 15 <= len(sentence.strip()) <= 600
    ]

    # 자동 생성 자막처럼 문장부호가 부족한 경우
    if len(sentences) <= 1 and len(text) > 400:
        chunk_size = 250

        sentences = [
            text[index:index + chunk_size].strip()
            for index in range(0, len(text), chunk_size)
            if len(text[index:index + chunk_size].strip()) >= 15
        ]

    return sentences


# =========================================================
# 4. 자막 파일 업로드 처리
# =========================================================

def decode_uploaded_file(uploaded_file):
    raw_data = uploaded_file.getvalue()

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "자막 파일의 문자 인코딩을 읽지 못했습니다."
    )


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
        raise ValueError(
            "업로드한 파일의 자막 텍스트가 부족합니다."
        )

    return subtitle_text


# =========================================================
# 5. YouTube URL 및 Data API
# =========================================================

def extract_video_id(url):
    if not url:
        return None

    url = url.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)

    hostname = (
        parsed.hostname or ""
    ).lower().replace("www.", "")

    if hostname == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if len(candidate) == 11 else None

    if hostname in {
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        if parsed.path == "/watch":
            candidate = parse_qs(
                parsed.query
            ).get("v", [None])[0]

            if candidate and len(candidate) == 11:
                return candidate

        path_parts = parsed.path.strip("/").split("/")

        if (
            len(path_parts) >= 2
            and path_parts[0] in {
                "shorts",
                "embed",
                "live",
            }
        ):
            candidate = path_parts[1]

            return (
                candidate
                if len(candidate) == 11
                else None
            )

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

    published_at = snippet.get(
        "publishedAt",
        "",
    )

    days = calculate_days_since_upload(
        published_at
    )

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get(
            "description",
            "",
        ),
        "channel_title": snippet.get(
            "channelTitle",
            "",
        ),
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


# =========================================================
# 6. 자막 수집
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_transcript(video_id):
    """
    입력 영상 한 개의 자막만 수집합니다.
    """

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
            "available_transcripts": (
                available_transcripts or []
            ),
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

    def priority(transcript):
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

    transcripts.sort(key=priority)

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
                snippet_text = getattr(
                    snippet,
                    "text",
                    "",
                )

                if snippet_text:
                    text_parts.append(
                        str(snippet_text).replace(
                            "\n",
                            " ",
                        )
                    )

            text = clean_text(
                " ".join(text_parts)
            )

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
                    available_transcripts=(
                        available_transcripts
                    ),
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
        if not getattr(
            transcript,
            "is_translatable",
            False,
        ):
            continue

        source_language = getattr(
            transcript,
            "language_code",
            "unknown",
        )

        for target_language in ["ko", "en"]:
            try:
                translated = transcript.translate(
                    target_language
                )

                fetched = translated.fetch()

                text_parts = []

                for snippet in fetched:
                    snippet_text = getattr(
                        snippet,
                        "text",
                        "",
                    )

                    if snippet_text:
                        text_parts.append(
                            str(snippet_text).replace(
                                "\n",
                                " ",
                            )
                        )

                text = clean_text(
                    " ".join(text_parts)
                )

                if len(
                    tokenize_mixed_text(text)
                ) >= 5:
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
                        available_transcripts=(
                            available_transcripts
                        ),
                    )

            except Exception as error:
                errors.append(
                    f"{source_language} → "
                    f"{target_language}: "
                    f"{type(error).__name__}: {error}"
                )

    return create_result(
        error_type="TranscriptFetchFailed",
        error_message="\n".join(errors),
        available_transcripts=available_transcripts,
    )


# =========================================================
# 7. 입력 데이터 준비
# =========================================================

def prepare_source_data(
    video_id,
    source_details,
    manual_transcript,
    uploaded_subtitle,
):
    """
    자막 사용 우선순위:
    1. 직접 입력
    2. 파일 업로드
    3. YouTube 자동 수집
    4. 제목·설명 대체
    """

    if manual_transcript.strip():
        return {
            "text": limit_text(manual_transcript),
            "label": "사용자가 직접 입력한 자막",
            "transcript_result": None,
        }

    if uploaded_subtitle is not None:
        return {
            "text": read_uploaded_subtitle(
                uploaded_subtitle
            ),
            "label": (
                f"업로드 자막: "
                f"{uploaded_subtitle.name}"
            ),
            "transcript_result": None,
        }

    transcript_result = get_transcript(video_id)

    if transcript_result["success"]:
        if transcript_result["translated"]:
            label = "YouTube 번역 자막"

        elif transcript_result["generated"]:
            label = "YouTube 자동 생성 자막"

        else:
            label = "YouTube 수동 등록 자막"

        return {
            "text": transcript_result["text"],
            "label": label,
            "transcript_result": transcript_result,
        }

    fallback_text = " ".join(
        [
            source_details["title"],
            source_details["title"],
            source_details["description"],
        ]
    )

    if len(
        tokenize_mixed_text(fallback_text)
    ) < 5:
        raise ValueError(
            "자막 자동 수집에 실패했고 영상 설명도 "
            "부족합니다. 자막을 직접 입력하거나 "
            "자막 파일을 업로드하세요."
        )

    return {
        "text": fallback_text,
        "label": "자막 수집 실패: 제목·설명 사용",
        "transcript_result": transcript_result,
    }


# =========================================================
# 8. TF-IDF 분석
# =========================================================

def analyze_tfidf(
    text,
    sentence_count=5,
    keyword_count=20,
):
    """
    TF-IDF 분석 결과만 생성합니다.
    추천 영상 계산과 분리되어 있습니다.
    """

    sentences = split_sentences(text)

    if not sentences:
        raise ValueError(
            "TF-IDF로 분석할 문장이 없습니다."
        )

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=MAX_TFIDF_FEATURES,
    )

    matrix = vectorizer.fit_transform(
        sentences
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    sentence_scores = np.asarray(
        matrix.sum(axis=1)
    ).ravel()

    top_count = min(
        sentence_count,
        len(sentences),
    )

    important_indices = np.argsort(
        sentence_scores
    )[::-1][:top_count]

    ordered_indices = sorted(
        important_indices.tolist()
    )

    summary = " ".join(
        sentences[index]
        for index in ordered_indices
    )

    sentence_df = pd.DataFrame(
        [
            {
                "중요도 순위": rank + 1,
                "원래 문장 순서": int(
                    index + 1
                ),
                "TF-IDF 문장 점수": round(
                    float(
                        sentence_scores[index]
                    ),
                    6,
                ),
                "문장": sentences[index],
            }
            for rank, index in enumerate(
                important_indices
            )
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
            "순위": range(
                1,
                len(keyword_indices) + 1,
            ),
            "단어 또는 구": keywords,
            "전체 TF-IDF 점수": [
                round(
                    float(
                        total_term_scores[index]
                    ),
                    6,
                )
                for index in keyword_indices
            ],
        }
    )

    sentence_feature_columns = [
        (
            f"tfidf_{index:04d}_"
            f"{feature.replace(' ', '_')}"
        )
        for index, feature in enumerate(
            feature_names
        )
    ]

    sentence_feature_df = pd.DataFrame(
        matrix.toarray(),
        columns=sentence_feature_columns,
    )

    sentence_feature_df.insert(
        0,
        "sentence",
        sentences,
    )

    sentence_feature_df.insert(
        1,
        "sentence_tfidf_score",
        sentence_scores,
    )

    return {
        "summary": summary,
        "keywords": keywords,
        "sentence_scores": sentence_df,
        "keyword_scores": keyword_df,
        "sentence_features": sentence_feature_df,
    }


# =========================================================
# 9. 검색어 생성 및 후보 영상 검색
# =========================================================

def build_search_query(
    source_title,
    keywords,
    custom_query="",
):
    if custom_query.strip():
        return clean_text(
            custom_query
        )[:100]

    title_tokens = tokenize_mixed_text(
        source_title
    )

    keyword_tokens = []

    for keyword in keywords:
        for token in keyword.split():
            token = clean_text(token)

            if (
                len(token) >= 2
                and token not in keyword_tokens
            ):
                keyword_tokens.append(token)

    query_tokens = []

    for token in (
        title_tokens[:3]
        + keyword_tokens[:3]
    ):
        if token not in query_tokens:
            query_tokens.append(token)

    return " ".join(query_tokens)[:100]


@st.cache_data(ttl=3600, show_spinner=False)
def search_candidate_videos(
    query,
    api_key,
    max_results=20,
):
    """
    후보 검색은 한 번만 수행합니다.
    코사인 유사도 정렬에는 검색 순위를 사용하지 않습니다.
    """

    youtube = get_youtube_client(api_key)

    response = (
        youtube.search()
        .list(
            part="snippet",
            q=query,
            type="video",
            maxResults=min(
                max_results,
                50,
            ),
            order="relevance",
            safeSearch="moderate",
        )
        .execute()
    )

    video_ids = [
        item["id"]["videoId"]
        for item in response.get(
            "items",
            [],
        )
        if item.get(
            "id",
            {},
        ).get("videoId")
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

    detail_items = {
        item.get("id", ""): item
        for item in detail_response.get(
            "items",
            [],
        )
    }

    candidates = []

    for video_id in video_ids:
        item = detail_items.get(video_id)

        if not item:
            continue

        snippet = item.get("snippet", {})
        statistics = item.get(
            "statistics",
            {},
        )

        title = snippet.get("title", "")
        description = snippet.get(
            "description",
            "",
        )

        candidate_text = clean_text(
            " ".join(
                [
                    title,
                    title,
                    title,
                    description,
                ]
            )
        )

        if len(
            tokenize_mixed_text(candidate_text)
        ) < 2:
            continue

        views = int(
            statistics.get(
                "viewCount",
                0,
            )
        )

        likes = int(
            statistics.get(
                "likeCount",
                0,
            )
        )

        published_at = snippet.get(
            "publishedAt",
            "",
        )

        days = calculate_days_since_upload(
            published_at
        )

        candidates.append(
            {
                "video_id": video_id,
                "title": title,
                "description": description,
                "comparison_text": candidate_text,
                "channel_title": snippet.get(
                    "channelTitle",
                    "",
                ),
                "published_at": published_at,
                "views": views,
                "likes": likes,
                "daily_views": views / days,
                "thumbnail": get_thumbnail(
                    snippet
                ),
                "url": (
                    "https://www.youtube.com/"
                    f"watch?v={video_id}"
                ),
            }
        )

    return candidates


# =========================================================
# 10. 코사인 유사도 추천
# =========================================================

def calculate_cosine_recommendations(
    source_title,
    tfidf_result,
    candidates,
):
    """
    추천 결과는 TF-IDF 벡터의 코사인 유사도로만 결정합니다.

    검색 순위, 조회수, 좋아요는 정렬 점수에 넣지 않습니다.
    """

    if not candidates:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    source_comparison_text = " ".join(
        [
            source_title,
            source_title,
            source_title,
            tfidf_result["summary"],
            " ".join(
                tfidf_result["keywords"][:15]
            ),
        ]
    )

    candidate_texts = [
        candidate["comparison_text"]
        for candidate in candidates
    ]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=MAX_TFIDF_FEATURES,
    )

    matrix = vectorizer.fit_transform(
        [source_comparison_text]
        + candidate_texts
    )

    source_vector = matrix[0:1]
    candidate_vectors = matrix[1:]

    similarities = cosine_similarity(
        source_vector,
        candidate_vectors,
    ).ravel()

    rows = []

    for index, candidate in enumerate(
        candidates
    ):
        row = candidate.copy()

        row["cosine_similarity"] = float(
            similarities[index]
        )

        rows.append(row)

    all_results = pd.DataFrame(rows)

    # 최소 기준을 통과한 영상만 추천합니다.
    recommendation_df = all_results[
        all_results["cosine_similarity"]
        >= MINIMUM_COSINE_SIMILARITY
    ].copy()

    recommendation_df = (
        recommendation_df.sort_values(
            by="cosine_similarity",
            ascending=False,
        ).reset_index(drop=True)
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    feature_columns = [
        (
            f"tfidf_{index:04d}_"
            f"{term.replace(' ', '_')}"
        )
        for index, term in enumerate(
            feature_names
        )
    ]

    feature_df = pd.DataFrame(
        candidate_vectors.toarray(),
        columns=feature_columns,
    )

    metadata_df = all_results[
        [
            "video_id",
            "title",
            "channel_title",
            "url",
            "views",
            "likes",
            "daily_views",
            "cosine_similarity",
        ]
    ].reset_index(drop=True)

    orange_df = pd.concat(
        [
            metadata_df,
            feature_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return recommendation_df, orange_df


def create_pair_tfidf_table(
    source_title,
    tfidf_result,
    candidate,
    maximum_terms=30,
):
    """
    입력 영상과 선택된 추천 영상의 TF-IDF 값을 표로 만듭니다.
    """

    source_text = " ".join(
        [
            source_title,
            source_title,
            source_title,
            tfidf_result["summary"],
            " ".join(
                tfidf_result["keywords"][:15]
            ),
        ]
    )

    comparison_text = candidate[
        "comparison_text"
    ]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        norm=None,
        use_idf=True,
        smooth_idf=True,
        max_features=MAX_TFIDF_FEATURES,
    )

    matrix = vectorizer.fit_transform(
        [
            source_text,
            comparison_text,
        ]
    )

    terms = (
        vectorizer.get_feature_names_out()
    )

    idf_values = vectorizer.idf_

    source_values = (
        matrix[0].toarray().ravel()
    )

    candidate_values = (
        matrix[1].toarray().ravel()
    )

    dataframe = pd.DataFrame(
        {
            "단어": terms,
            "IDF": idf_values,
            "입력 영상 TF-IDF": source_values,
            "후보 영상 TF-IDF": candidate_values,
            "두 영상 TF-IDF 합계": (
                source_values
                + candidate_values
            ),
        }
    )

    return dataframe.sort_values(
        by="두 영상 TF-IDF 합계",
        ascending=False,
    ).head(maximum_terms)


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# =========================================================
# 11. 세션 상태
# =========================================================

DEFAULT_SESSION_VALUES = {
    "source_details": None,
    "source_text": "",
    "source_label": "",
    "tfidf_result": None,
    "recommendation_df": None,
    "orange_df": None,
    "search_query_used": "",
    "candidate_lookup": {},
}

for key, value in DEFAULT_SESSION_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_analysis_results():
    st.session_state.source_details = None
    st.session_state.source_text = ""
    st.session_state.source_label = ""
    st.session_state.tfidf_result = None
    st.session_state.recommendation_df = None
    st.session_state.orange_df = None
    st.session_state.search_query_used = ""
    st.session_state.candidate_lookup = {}


# =========================================================
# 12. 공통 분석 준비
# =========================================================

def load_source_and_tfidf(
    youtube_url,
    api_key,
    manual_transcript,
    uploaded_subtitle,
    sentence_count,
):
    video_id = extract_video_id(youtube_url)

    if not video_id:
        raise ValueError(
            "올바른 YouTube URL을 입력하세요."
        )

    source_details = get_video_details(
        video_id,
        api_key,
    )

    if not source_details:
        raise ValueError(
            "입력 영상 정보를 가져오지 못했습니다."
        )

    source_data = prepare_source_data(
        video_id=video_id,
        source_details=source_details,
        manual_transcript=manual_transcript,
        uploaded_subtitle=uploaded_subtitle,
    )

    tfidf_result = analyze_tfidf(
        source_data["text"],
        sentence_count=sentence_count,
    )

    st.session_state.source_details = (
        source_details
    )

    st.session_state.source_text = (
        source_data["text"]
    )

    st.session_state.source_label = (
        source_data["label"]
    )

    st.session_state.tfidf_result = (
        tfidf_result
    )

    return (
        video_id,
        source_details,
        tfidf_result,
    )


# =========================================================
# 13. 화면 구성
# =========================================================

st.title(
    "🎬 YouTube TF-IDF 분석 및 코사인 유사도 추천"
)

st.write(
    "TF-IDF 자막 분석과 코사인 유사도 추천을 "
    "서로 다른 버튼과 결과 영역으로 분리했습니다."
)

with st.expander(
    "📐 분석 구조 확인"
):
    st.markdown(
        r"""
### 1. TF-IDF 분석

입력 영상의 자막을 문장 단위로 나누고 단어의
TF-IDF 값을 계산합니다.

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

TF-IDF 결과로 다음을 출력합니다.

- 핵심 문장 요약
- 핵심 단어 및 구
- 문장별 TF-IDF 중요도
- TF-IDF CSV

### 2. 코사인 유사도 추천

입력 영상의 제목·TF-IDF 요약·핵심 키워드와
후보 영상의 제목·설명을 같은 TF-IDF 공간에 배치합니다.

\[
CosineSimilarity(A,B)
=
\frac{A\cdot B}
{\|A\|\|B\|}
\]

추천 영상은 코사인 유사도 내림차순으로만 정렬합니다.
조회수, 좋아요, 검색 순위는 추천 점수에 포함하지 않습니다.
        """
    )

if st.sidebar.button(
    "🗑️ 캐시 및 결과 초기화",
    use_container_width=True,
):
    st.cache_data.clear()
    reset_analysis_results()

    st.sidebar.success(
        "캐시와 분석 결과를 초기화했습니다."
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
        "추천 정렬 기준\n\n"
        "- TF-IDF 코사인 유사도만 사용\n"
        "- 최소 유사도: 0.08\n"
        "- 조회수·좋아요는 참고 정보"
    )

api_key = get_api_key()

if not api_key:
    st.warning(
        "Streamlit Secrets에 "
        "YOUTUBE_API_KEY를 등록하세요."
    )

youtube_url = st.text_input(
    "분석할 YouTube 영상 URL",
    placeholder=(
        "https://www.youtube.com/"
        "watch?v=..."
    ),
)

custom_search_query = st.text_input(
    "후보 영상 검색어(선택)",
    placeholder=(
        "예: 생성형 인공지능 교육 활용"
    ),
    help=(
        "비워 두면 영상 제목과 TF-IDF "
        "핵심 키워드로 자동 생성합니다."
    ),
)

with st.expander(
    "자막 자동 수집 실패 시 대체 입력"
):
    uploaded_subtitle = st.file_uploader(
        "TXT, SRT 또는 VTT 자막 파일",
        type=[
            "txt",
            "srt",
            "vtt",
        ],
    )

    manual_transcript = st.text_area(
        "또는 자막을 직접 붙여 넣으세요.",
        height=150,
    )

button_column1, button_column2, button_column3 = (
    st.columns(3)
)

with button_column1:
    diagnostic_button = st.button(
        "🔍 자막 진단",
        use_container_width=True,
    )

with button_column2:
    tfidf_button = st.button(
        "📊 TF-IDF 자막 분석",
        type="primary",
        use_container_width=True,
    )

with button_column3:
    cosine_button = st.button(
        "🎯 코사인 유사도 추천",
        use_container_width=True,
    )


# =========================================================
# 14. 자막 진단 실행
# =========================================================

if diagnostic_button:
    diagnostic_video_id = extract_video_id(
        youtube_url
    )

    if not diagnostic_video_id:
        st.error(
            "올바른 YouTube URL을 입력하세요."
        )

    else:
        with st.spinner(
            "자막 상태를 확인하고 있습니다..."
        ):
            diagnostic_result = get_transcript(
                diagnostic_video_id
            )

        if diagnostic_result["success"]:
            st.success(
                "자막을 정상적으로 가져왔습니다."
            )

            metric1, metric2, metric3 = (
                st.columns(3)
            )

            metric1.metric(
                "언어",
                diagnostic_result["language"],
            )

            metric2.metric(
                "자막 유형",
                (
                    "자동 생성"
                    if diagnostic_result[
                        "generated"
                    ]
                    else "수동 등록"
                ),
            )

            metric3.metric(
                "번역 여부",
                (
                    "번역"
                    if diagnostic_result[
                        "translated"
                    ]
                    else "원본"
                ),
            )

            st.text_area(
                "자막 미리보기",
                diagnostic_result[
                    "text"
                ][:2000],
                height=250,
            )

        else:
            st.error(
                "YouTube 자막 자동 수집에 실패했습니다."
            )

            st.write(
                "오류 유형: "
                f"`{diagnostic_result['error_type']}`"
            )

            st.code(
                diagnostic_result[
                    "error_message"
                ]
                or "오류 메시지가 없습니다."
            )

            st.info(
                "자막이 없는 것과는 다를 수 있습니다. "
                "직접 입력 또는 자막 파일 업로드를 이용할 수 있습니다."
            )


# =========================================================
# 15. TF-IDF 분석 버튼 실행
# =========================================================

if tfidf_button:
    if not api_key:
        st.error(
            "YouTube Data API 키를 설정하세요."
        )

    else:
        try:
            with st.spinner(
                "영상 정보와 자막을 불러와 "
                "TF-IDF를 분석하고 있습니다..."
            ):
                load_source_and_tfidf(
                    youtube_url=youtube_url,
                    api_key=api_key,
                    manual_transcript=manual_transcript,
                    uploaded_subtitle=uploaded_subtitle,
                    sentence_count=(
                        summary_sentence_count
                    ),
                )

            # 새로운 TF-IDF 분석을 했으므로
            # 이전 추천 결과는 삭제합니다.
            st.session_state.recommendation_df = None
            st.session_state.orange_df = None
            st.session_state.search_query_used = ""
            st.session_state.candidate_lookup = {}

            st.success(
                "TF-IDF 자막 분석이 완료되었습니다."
            )

        except Exception as error:
            st.error(
                "TF-IDF 분석 중 오류가 발생했습니다."
            )

            st.write(
                f"오류 유형: "
                f"`{type(error).__name__}`"
            )

            st.write(
                f"오류 내용: {error}"
            )

            with st.expander(
                "개발자용 상세 오류"
            ):
                st.code(
                    traceback.format_exc()
                )


# =========================================================
# 16. 코사인 유사도 추천 버튼 실행
# =========================================================

if cosine_button:
    if not api_key:
        st.error(
            "YouTube Data API 키를 설정하세요."
        )

    else:
        try:
            # TF-IDF 분석을 아직 하지 않았다면
            # 추천 실행 시 한 번 자동 분석합니다.
            (
                video_id,
                source_details,
                tfidf_result,
            ) = load_source_and_tfidf(
                youtube_url=youtube_url,
                api_key=api_key,
                manual_transcript=manual_transcript,
                uploaded_subtitle=uploaded_subtitle,
                sentence_count=(
                    summary_sentence_count
                ),
            )

            search_query = build_search_query(
                source_title=(
                    source_details["title"]
                ),
                keywords=(
                    tfidf_result["keywords"]
                ),
                custom_query=(
                    custom_search_query
                ),
            )

            if not search_query:
                raise ValueError(
                    "후보 검색어를 만들지 못했습니다."
                )

            with st.spinner(
                "후보 영상을 검색하고 코사인 "
                "유사도를 계산하고 있습니다..."
            ):
                candidates = (
                    search_candidate_videos(
                        query=search_query,
                        api_key=api_key,
                        max_results=(
                            candidate_count
                        ),
                    )
                )

            candidates = [
                candidate
                for candidate in candidates
                if candidate["video_id"]
                != video_id
            ]

            if not candidates:
                raise ValueError(
                    "입력 영상 이외의 후보 영상을 "
                    "찾지 못했습니다."
                )

            (
                recommendation_df,
                orange_df,
            ) = calculate_cosine_recommendations(
                source_title=(
                    source_details["title"]
                ),
                tfidf_result=tfidf_result,
                candidates=candidates,
            )

            st.session_state.recommendation_df = (
                recommendation_df
            )

            st.session_state.orange_df = (
                orange_df
            )

            st.session_state.search_query_used = (
                search_query
            )

            st.session_state.candidate_lookup = {
                candidate["video_id"]: candidate
                for candidate in candidates
            }

            if recommendation_df.empty:
                st.warning(
                    "최소 코사인 유사도 기준을 "
                    "통과한 후보 영상이 없습니다."
                )

                st.info(
                    "관련성이 거의 없는 영상을 "
                    "강제로 추천하지 않았습니다. "
                    "후보 영상 검색어를 짧고 "
                    "명확하게 입력해 보세요."
                )

            else:
                st.success(
                    "코사인 유사도 추천이 완료되었습니다."
                )

        except Exception as error:
            st.error(
                "코사인 유사도 추천 중 오류가 발생했습니다."
            )

            st.write(
                f"오류 유형: "
                f"`{type(error).__name__}`"
            )

            st.write(
                f"오류 내용: {error}"
            )

            with st.expander(
                "개발자용 상세 오류"
            ):
                st.code(
                    traceback.format_exc()
                )


# =========================================================
# 17. 입력 영상 공통 정보 출력
# =========================================================

if st.session_state.source_details is not None:
    source_details = (
        st.session_state.source_details
    )

    st.divider()

    st.subheader("입력 영상")

    video_column, information_column = (
        st.columns([1, 2])
    )

    with video_column:
        st.video(
            source_details["url"]
        )

    with information_column:
        st.markdown(
            f"### {source_details['title']}"
        )

        st.write(
            "채널: "
            f"{source_details['channel_title']}"
        )

        st.write(
            "분석 데이터: "
            f"{st.session_state.source_label}"
        )

        st.write(
            f"조회수: "
            f"{source_details['views']:,}"
        )

        st.write(
            f"좋아요: "
            f"{source_details['likes']:,}"
        )


# =========================================================
# 18. TF-IDF 분석 결과 출력
# =========================================================

if st.session_state.tfidf_result is not None:
    tfidf_result = (
        st.session_state.tfidf_result
    )

    st.divider()

    st.header("📊 TF-IDF 자막 분석 결과")

    st.subheader("핵심 내용 요약")

    st.write(
        tfidf_result["summary"]
    )

    st.subheader(
        "핵심 키워드 및 TF-IDF 점수"
    )

    st.dataframe(
        tfidf_result["keyword_scores"],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "문장별 TF-IDF 중요도"
    ):
        st.dataframe(
            tfidf_result["sentence_scores"],
            use_container_width=True,
            hide_index=True,
        )

    download_column1, download_column2 = (
        st.columns(2)
    )

    with download_column1:
        st.download_button(
            "TF-IDF 키워드 CSV",
            data=dataframe_to_csv_bytes(
                tfidf_result[
                    "keyword_scores"
                ]
            ),
            file_name=(
                "tfidf_keyword_results.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with download_column2:
        st.download_button(
            "TF-IDF 문장 벡터 CSV",
            data=dataframe_to_csv_bytes(
                tfidf_result[
                    "sentence_features"
                ]
            ),
            file_name=(
                "tfidf_sentence_features.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# 19. 코사인 유사도 결과 출력
# =========================================================

recommendation_df = (
    st.session_state.recommendation_df
)

if (
    recommendation_df is not None
    and not recommendation_df.empty
):
    st.divider()

    st.header(
        "🎯 코사인 유사도 추천 결과"
    )

    st.write(
        "후보 영상 검색어: "
        f"`{st.session_state.search_query_used}`"
    )

    st.caption(
        "추천 순서는 TF-IDF 벡터 사이의 "
        "코사인 유사도만으로 결정됩니다."
    )

    shown_df = recommendation_df.head(
        recommendation_count
    )

    for rank, (_, row) in enumerate(
        shown_df.iterrows(),
        start=1,
    ):
        st.markdown("---")

        image_column, result_column = (
            st.columns([1, 3])
        )

        with image_column:
            if row["thumbnail"]:
                st.image(
                    row["thumbnail"],
                    use_container_width=True,
                )

        with result_column:
            st.markdown(
                f"### {rank}. "
                f"[{row['title']}]"
                f"({row['url']})"
            )

            st.write(
                "채널: "
                f"{row['channel_title']}"
            )

            st.metric(
                "코사인 유사도",
                (
                    f"{row['cosine_similarity']:.4f}"
                ),
            )

            st.write(
                f"조회수: "
                f"{int(row['views']):,} | "
                f"좋아요: "
                f"{int(row['likes']):,}"
            )

    st.subheader(
        "추천 영상별 코사인 유사도"
    )

    chart_df = shown_df[
        [
            "title",
            "cosine_similarity",
        ]
    ].set_index("title")

    st.bar_chart(chart_df)

    st.subheader(
        "추천 1위 영상과의 TF-IDF 비교"
    )

    first_row = shown_df.iloc[0]

    first_candidate = (
        st.session_state.candidate_lookup[
            first_row["video_id"]
        ]
    )

    pair_tfidf_df = create_pair_tfidf_table(
        source_title=(
            st.session_state.source_details[
                "title"
            ]
        ),
        tfidf_result=(
            st.session_state.tfidf_result
        ),
        candidate=first_candidate,
    )

    st.dataframe(
        pair_tfidf_df,
        use_container_width=True,
        hide_index=True,
    )

    download_column1, download_column2, download_column3 = (
        st.columns(3)
    )

    with download_column1:
        st.download_button(
            "코사인 추천 결과 CSV",
            data=dataframe_to_csv_bytes(
                recommendation_df
            ),
            file_name=(
                "cosine_recommendations.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with download_column2:
        st.download_button(
            "Orange3 후보 TF-IDF CSV",
            data=dataframe_to_csv_bytes(
                st.session_state.orange_df
            ),
            file_name=(
                "orange_candidate_tfidf.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with download_column3:
        st.download_button(
            "추천 1위 TF-IDF 비교 CSV",
            data=dataframe_to_csv_bytes(
                pair_tfidf_df
            ),
            file_name=(
                "top_recommendation_tfidf.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )
