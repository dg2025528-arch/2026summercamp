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
from sklearn.preprocessing import MinMaxScaler
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# 1. 페이지 및 모델 설정
# =========================================================

st.set_page_config(
    page_title="YouTube TF-IDF 영상 추천",
    page_icon="🎬",
    layout="wide",
)

SEARCH_RELEVANCE_WEIGHT = 0.45
TITLE_SIMILARITY_WEIGHT = 0.45
POPULARITY_WEIGHT = 0.10

MINIMUM_TITLE_SIMILARITY = 0.08
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
# 4. 업로드 자막 처리
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
# 5. URL 및 YouTube Data API
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
    published_at = snippet.get("publishedAt", "")
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
# 6. 입력 영상 자막 수집
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

                text = clean_text(
                    " ".join(
                        str(
                            getattr(
                                snippet,
                                "text",
                                "",
                            )
                        ).replace("\n", " ")
                        for snippet in fetched
                        if getattr(
                            snippet,
                            "text",
                            "",
                        )
                    )
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
                    f"{type(error).__name__}: "
                    f"{error}"
                )

    return create_result(
        error_type="TranscriptFetchFailed",
        error_message="\n".join(errors),
        available_transcripts=available_transcripts,
    )


# =========================================================
# 7. TF-IDF 요약 및 키워드
# =========================================================

def summarize_with_tfidf(
    text,
    sentence_count=5,
    keyword_count=20,
):
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

    matrix = vectorizer.fit_transform(sentences)
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
                "원래 순서": int(index + 1),
                "TF-IDF 문장 점수": round(
                    float(
                        sentence_scores[index]
                    ),
                    4,
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
            "TF-IDF 점수": [
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

    return {
        "summary": summary,
        "keywords": keywords,
        "sentence_scores": sentence_df,
        "keyword_scores": keyword_df,
    }


# =========================================================
# 8. 검색어 생성 및 후보 검색
# =========================================================

def build_search_queries(
    source_title,
    keywords,
):
    single_word_keywords = []

    for keyword in keywords:
        keyword = clean_text(keyword)

        for part in keyword.split():
            if (
                len(part) >= 2
                and part
                not in single_word_keywords
            ):
                single_word_keywords.append(part)

    title_tokens = tokenize_mixed_text(
        source_title
    )

    unique_title_tokens = []

    for token in title_tokens:
        if token not in unique_title_tokens:
            unique_title_tokens.append(token)

    queries = []

    if unique_title_tokens:
        queries.append(
            " ".join(
                unique_title_tokens[:4]
            )
        )

    if single_word_keywords:
        queries.append(
            " ".join(
                single_word_keywords[:4]
            )
        )

    mixed_query = " ".join(
        unique_title_tokens[:2]
        + single_word_keywords[:2]
    ).strip()

    if mixed_query:
        queries.append(mixed_query)

    if unique_title_tokens:
        queries.append(
            " ".join(
                unique_title_tokens[:2]
            )
        )

    unique_queries = []

    for query in queries:
        query = clean_text(query)[:100]

        if (
            query
            and query not in unique_queries
        ):
            unique_queries.append(query)

    return unique_queries


@st.cache_data(ttl=3600, show_spinner=False)
def search_candidate_videos(
    queries,
    api_key,
    max_results=20,
):
    youtube = get_youtube_client(api_key)

    if isinstance(queries, str):
        queries = [queries]

    collected_video_ids = []
    search_log = []

    # API 할당량을 줄이기 위해 최대 2개의 검색어만 사용합니다.
    for query in queries[:2]:
        if not query.strip():
            continue

        try:
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

            current_ids = [
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

            search_log.append(
                {
                    "검색어": query,
                    "결과 수": len(
                        current_ids
                    ),
                    "오류": "",
                }
            )

            for video_id in current_ids:
                if (
                    video_id
                    not in collected_video_ids
                ):
                    collected_video_ids.append(
                        video_id
                    )

                if (
                    len(collected_video_ids)
                    >= max_results
                ):
                    break

            if (
                len(collected_video_ids)
                >= max_results
            ):
                break

        except Exception as error:
            search_log.append(
                {
                    "검색어": query,
                    "결과 수": 0,
                    "오류": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

    if not collected_video_ids:
        return {
            "candidates": [],
            "search_log": search_log,
        }

    detail_response = (
        youtube.videos()
        .list(
            part="snippet,statistics",
            id=",".join(
                collected_video_ids[:50]
            ),
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

    for search_rank, video_id in enumerate(
        collected_video_ids,
        start=1,
    ):
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

        if (
            not clean_text(title)
            and not clean_text(description)
        ):
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
                "search_rank": search_rank,
            }
        )

    return {
        "candidates": candidates,
        "search_log": search_log,
    }


# =========================================================
# 9. 추천 모델
# =========================================================

def normalize_log_values(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return values

    logged = np.log1p(
        values
    ).reshape(-1, 1)

    if (
        len(values) == 1
        or np.allclose(
            logged,
            logged[0],
        )
    ):
        return np.zeros(len(values))

    return (
        MinMaxScaler()
        .fit_transform(logged)
        .ravel()
    )


def create_orange_title_features(
    title_matrix,
    vectorizer,
    all_results,
):
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
        title_matrix.toarray(),
        columns=feature_columns,
    )

    metadata_columns = [
        "video_id",
        "title",
        "channel_title",
        "url",
        "search_rank",
        "title_similarity",
        "search_relevance_score",
        "views",
        "likes",
        "daily_views",
        "popularity_score",
        "final_score",
    ]

    metadata_df = all_results[
        metadata_columns
    ].reset_index(drop=True)

    return pd.concat(
        [
            metadata_df,
            feature_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


def calculate_recommendations(
    source_title,
    source_keywords,
    candidates,
):
    if not candidates:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    source_query_text = " ".join(
        [
            source_title,
            source_title,
            " ".join(
                source_keywords[:10]
            ),
        ]
    )

    candidate_title_texts = [
        " ".join(
            [
                candidate["title"],
                candidate["title"],
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
        max_features=800,
    )

    title_matrix = (
        title_vectorizer.fit_transform(
            [source_query_text]
            + candidate_title_texts
        )
    )

    title_similarities = (
        cosine_similarity(
            title_matrix[0:1],
            title_matrix[1:],
        ).ravel()
    )

    search_ranks = np.array(
        [
            candidate.get(
                "search_rank",
                index + 1,
            )
            for index, candidate in enumerate(
                candidates
            )
        ],
        dtype=float,
    )

    if len(search_ranks) == 1:
        search_relevance_scores = (
            np.ones(1)
        )

    else:
        maximum_rank = float(
            np.max(search_ranks)
        )

        search_relevance_scores = (
            maximum_rank - search_ranks
        ) / max(
            maximum_rank - 1.0,
            1.0,
        )

    normalized_views = normalize_log_values(
        [
            candidate["daily_views"]
            for candidate in candidates
        ]
    )

    normalized_likes = normalize_log_values(
        [
            candidate["likes"]
            for candidate in candidates
        ]
    )

    popularity_scores = (
        0.6 * normalized_views
        + 0.4 * normalized_likes
    )

    final_scores = (
        SEARCH_RELEVANCE_WEIGHT
        * search_relevance_scores
        + TITLE_SIMILARITY_WEIGHT
        * title_similarities
        + POPULARITY_WEIGHT
        * popularity_scores
    )

    rows = []

    for index, candidate in enumerate(
        candidates
    ):
        row = candidate.copy()

        row["title_similarity"] = float(
            title_similarities[index]
        )

        row[
            "search_relevance_score"
        ] = float(
            search_relevance_scores[index]
        )

        row["popularity_score"] = float(
            popularity_scores[index]
        )

        row["final_score"] = float(
            final_scores[index]
        )

        rows.append(row)

    all_results = pd.DataFrame(rows)

    orange_df = create_orange_title_features(
        title_matrix=title_matrix[1:],
        vectorizer=title_vectorizer,
        all_results=all_results,
    )

    # 기준 미달 결과는 강제로 추천하지 않습니다.
    results = all_results[
        all_results["title_similarity"]
        >= MINIMUM_TITLE_SIMILARITY
    ].copy()

    if results.empty:
        return results, orange_df

    results = results.sort_values(
        by=[
            "final_score",
            "title_similarity",
            "search_rank",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return results, orange_df


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# =========================================================
# 10. 화면 구성
# =========================================================

st.title(
    "🎬 YouTube TF-IDF 자막 분석 및 영상 추천"
)

st.write(
    "TF-IDF로 입력 영상의 자막을 요약하고 핵심 키워드를 "
    "추출합니다. 추천은 YouTube 검색 관련성, 제목·키워드 "
    "유사도, 인기도를 결합하여 계산합니다."
)

if st.sidebar.button(
    "🗑️ 캐시 삭제",
    use_container_width=True,
):
    st.cache_data.clear()
    st.sidebar.success(
        "캐시를 삭제했습니다."
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
        "추천 모델 고정값\n\n"
        "- YouTube 검색 관련성: 45%\n"
        "- 제목·키워드 유사도: 45%\n"
        "- 인기도: 10%\n"
        "- 최소 제목 유사도: 0.08"
    )

with st.expander(
    "📐 TF-IDF 및 추천 점수 계산 원리"
):
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

### 최종 추천 점수

\[
FinalScore
=
0.45\times SearchRelevance
+
0.45\times TitleSimilarity
+
0.10\times Popularity
\]

제목·핵심 키워드 TF-IDF 유사도가 0.08 미만인 영상은
관련 영상으로 표시하지 않습니다.
        """
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
    "관련 영상 검색어 직접 입력(선택)",
    placeholder=(
        "예: 생성형 인공지능 교육 활용"
    ),
    help=(
        "비워 두면 제목과 TF-IDF 핵심 "
        "키워드로 자동 생성합니다."
    ),
)

st.subheader("자막 대체 입력")

uploaded_subtitle = st.file_uploader(
    "자동 자막 수집 실패 시 "
    "TXT, SRT 또는 VTT 파일 업로드",
    type=[
        "txt",
        "srt",
        "vtt",
    ],
)

manual_transcript = st.text_area(
    "또는 자막을 직접 붙여 넣으세요.",
    height=120,
)

button_column1, button_column2 = (
    st.columns(2)
)

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
# 11. 자막 진단
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
            "입력 영상 자막을 확인하고 있습니다..."
        ):
            diagnostic_result = get_transcript(
                diagnostic_video_id
            )

        if diagnostic_result["success"]:
            st.success(
                "자막을 정상적으로 가져왔습니다."
            )

            column1, column2, column3 = (
                st.columns(3)
            )

            column1.metric(
                "언어",
                diagnostic_result[
                    "language"
                ],
            )

            column2.metric(
                "자막 유형",
                (
                    "자동 생성"
                    if diagnostic_result[
                        "generated"
                    ]
                    else "수동 등록"
                ),
            )

            column3.metric(
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

            if diagnostic_result[
                "available_transcripts"
            ]:
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
                diagnostic_result[
                    "error_message"
                ]
                or "오류 메시지가 없습니다."
            )


# =========================================================
# 12. 분석 및 추천
# =========================================================

if analyze_button:
    if not api_key:
        st.error(
            "YouTube Data API 키를 설정하세요."
        )
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error(
            "올바른 YouTube URL을 입력하세요."
        )
        st.stop()

    try:
        with st.spinner(
            "입력 영상 정보를 가져오는 중입니다..."
        ):
            source_details = get_video_details(
                video_id,
                api_key,
            )

        if not source_details:
            st.error(
                "영상 정보를 가져오지 못했습니다."
            )
            st.stop()

        source_content = ""
        source_data_label = ""

        # 사용자가 제공한 자막을 우선합니다.
        if manual_transcript.strip():
            source_content = limit_text(
                manual_transcript
            )

            source_data_label = (
                "사용자가 직접 입력한 자막"
            )

        elif uploaded_subtitle is not None:
            source_content = (
                read_uploaded_subtitle(
                    uploaded_subtitle
                )
            )

            source_data_label = (
                "업로드 자막: "
                f"{uploaded_subtitle.name}"
            )

        else:
            with st.spinner(
                "입력 영상 자막을 가져오는 중입니다..."
            ):
                transcript_result = (
                    get_transcript(video_id)
                )

            if transcript_result["success"]:
                source_content = (
                    transcript_result["text"]
                )

                if transcript_result[
                    "translated"
                ]:
                    source_data_label = (
                        "YouTube 번역 자막"
                    )

                elif transcript_result[
                    "generated"
                ]:
                    source_data_label = (
                        "YouTube 자동 생성 자막"
                    )

                else:
                    source_data_label = (
                        "YouTube 수동 등록 자막"
                    )

            else:
                fallback_text = " ".join(
                    [
                        source_details[
                            "title"
                        ],
                        source_details[
                            "title"
                        ],
                        source_details[
                            "description"
                        ],
                    ]
                )

                if len(
                    tokenize_mixed_text(
                        fallback_text
                    )
                ) < 5:
                    st.error(
                        "자막 자동 수집에 실패했고 "
                        "영상 설명도 부족합니다. "
                        "자막을 직접 입력하거나 "
                        "파일을 업로드하세요."
                    )

                    st.write(
                        "자막 오류: "
                        f"`{transcript_result['error_type']}`"
                    )

                    st.stop()

                source_content = fallback_text
                source_data_label = (
                    "자막 수집 실패: 제목·설명 사용"
                )

                st.warning(
                    "자막 자동 수집에 실패하여 "
                    "제목과 설명으로 분석합니다."
                )

        with st.spinner(
            "TF-IDF로 자막을 분석 중입니다..."
        ):
            summary_result = (
                summarize_with_tfidf(
                    source_content,
                    sentence_count=(
                        summary_sentence_count
                    ),
                )
            )

        search_keywords = (
            summary_result["keywords"][:10]
        )

        search_queries = build_search_queries(
            source_details["title"],
            search_keywords,
        )

        if custom_search_query.strip():
            search_queries.insert(
                0,
                clean_text(
                    custom_search_query
                )[:100],
            )

            search_queries = list(
                dict.fromkeys(search_queries)
            )

        if not search_queries:
            st.error(
                "검색어를 생성하지 못했습니다."
            )
            st.stop()

        with st.spinner(
            "관련 영상 후보를 검색 중입니다..."
        ):
            search_result = (
                search_candidate_videos(
                    search_queries,
                    api_key,
                    candidate_count,
                )
            )

        candidates = search_result[
            "candidates"
        ]

        search_log = search_result[
            "search_log"
        ]

        candidates = [
            candidate
            for candidate in candidates
            if candidate["video_id"]
            != video_id
        ]

        with st.expander(
            "🔎 후보 영상 검색 과정"
        ):
            st.write("사용된 검색어")

            for index, query in enumerate(
                search_queries[:2],
                start=1,
            ):
                st.write(
                    f"{index}. `{query}`"
                )

            if search_log:
                st.dataframe(
                    pd.DataFrame(
                        search_log
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.write(
                "입력 영상 제외 후 후보 수: "
                f"{len(candidates)}개"
            )

        if not candidates:
            st.error(
                "입력 영상 이외의 후보 영상을 "
                "찾지 못했습니다."
            )
            st.stop()

        with st.spinner(
            "추천 점수를 계산 중입니다..."
        ):
            results_df, orange_df = (
                calculate_recommendations(
                    source_title=(
                        source_details["title"]
                    ),
                    source_keywords=(
                        summary_result[
                            "keywords"
                        ]
                    ),
                    candidates=candidates,
                )
            )

        if results_df.empty:
            st.warning(
                "제목·키워드 유사도 기준을 "
                "통과한 관련 영상이 없습니다."
            )

            st.info(
                "관련성이 거의 없는 영상을 "
                "강제로 추천하지 않았습니다. "
                "관련 영상 검색어를 더 짧고 "
                "명확하게 입력해 보세요."
            )

            st.write(
                "적용된 최소 제목 유사도: "
                f"`{MINIMUM_TITLE_SIMILARITY}`"
            )

            st.stop()

        st.success(
            "분석이 완료되었습니다."
        )

        st.subheader("1. 입력 영상")

        video_column, info_column = (
            st.columns([1, 2])
        )

        with video_column:
            st.video(
                source_details["url"]
            )

        with info_column:
            st.markdown(
                f"### {source_details['title']}"
            )

            st.write(
                "채널: "
                f"{source_details['channel_title']}"
            )

            st.write(
                "분석 데이터: "
                f"{source_data_label}"
            )

            st.write(
                "조회수: "
                f"{source_details['views']:,}"
            )

            st.write(
                "좋아요: "
                f"{source_details['likes']:,}"
            )

        st.subheader(
            "2. TF-IDF 핵심 내용 요약"
        )

        st.write(
            summary_result["summary"]
        )

        st.subheader(
            "3. TF-IDF 핵심 키워드"
        )

        st.dataframe(
            summary_result[
                "keyword_scores"
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "문장별 TF-IDF 중요도"
        ):
            st.dataframe(
                summary_result[
                    "sentence_scores"
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(
            "4. 관련 영상 추천"
        )

        shown_df = results_df.head(
            recommendation_count
        )

        for index, row in shown_df.iterrows():
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
                    f"### {index + 1}. "
                    f"[{row['title']}]"
                    f"({row['url']})"
                )

                st.write(
                    "채널: "
                    f"{row['channel_title']}"
                )

                metric1, metric2, metric3 = (
                    st.columns(3)
                )

                metric1.metric(
                    "제목·키워드 유사도",
                    (
                        f"{row['title_similarity']:.3f}"
                    ),
                )

                metric2.metric(
                    "검색 관련성",
                    (
                        f"{row['search_relevance_score']:.3f}"
                    ),
                )

                metric3.metric(
                    "최종 점수",
                    (
                        f"{row['final_score']:.3f}"
                    ),
                )

                st.write(
                    "YouTube 검색 순위: "
                    f"{int(row['search_rank'])}위 | "
                    "조회수: "
                    f"{int(row['views']):,} | "
                    "좋아요: "
                    f"{int(row['likes']):,}"
                )

        st.subheader(
            "5. 추천 결과 비교"
        )

        chart_df = shown_df[
            [
                "title",
                "title_similarity",
                "search_relevance_score",
                "final_score",
            ]
        ].set_index("title")

        st.bar_chart(chart_df)

        st.subheader(
            "6. CSV 다운로드"
        )

        column1, column2 = st.columns(2)

        with column1:
            st.download_button(
                "추천 결과 CSV",
                data=dataframe_to_csv_bytes(
                    results_df
                ),
                file_name=(
                    "recommendation_results.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with column2:
            st.download_button(
                "Orange3 TF-IDF CSV",
                data=dataframe_to_csv_bytes(
                    orange_df
                ),
                file_name=(
                    "orange_tfidf_features.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as error:
        st.error(
            "분석 중 오류가 발생했습니다."
        )

        st.write(
            "오류 유형: "
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
