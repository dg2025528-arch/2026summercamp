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
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="YouTube TF-IDF 및 코사인 유사도 분석",
    page_icon="🎬",
    layout="wide",
)

MAX_TRANSCRIPT_CHARACTERS = 30000
MAX_TFIDF_FEATURES = 1500

# 절대값은 거의 0인 결과만 제외합니다.
ABSOLUTE_MINIMUM_SIMILARITY = 0.001

# 최고 후보의 30% 이상인 후보를 상대 추천합니다.
RELATIVE_MINIMUM_SIMILARITY = 0.30

# 최고 유사도가 이보다 낮으면 품질 경고를 표시합니다.
LOW_QUALITY_WARNING = 0.03


# =========================================================
# 2. 한국어 불용어
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
    if text is None:
        return ""

    text = str(text)

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


def limit_text(text):
    return clean_text(text)[:MAX_TRANSCRIPT_CHARACTERS]


def tokenize_mixed_text(text):
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

        if len(token) < 2:
            continue

        filtered_tokens.append(token)

    return filtered_tokens


def split_sentences(text):
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+|(?<=[다요죠])\s+(?=[가-힣A-Za-z0-9])",
        text,
    )

    valid_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if 15 <= len(sentence) <= 600:
            valid_sentences.append(sentence)

    # 자동 자막에 문장부호가 없는 경우 글자 수로 나눕니다.
    if len(valid_sentences) <= 1 and len(text) > 300:
        valid_sentences = []

        for index in range(0, len(text), 250):
            chunk = text[index:index + 250].strip()

            if len(chunk) >= 15:
                valid_sentences.append(chunk)

    return valid_sentences


# =========================================================
# 4. 자막 파일 처리
# =========================================================

def decode_uploaded_file(uploaded_file):
    raw_data = uploaded_file.getvalue()

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]

    for encoding in encodings:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "자막 파일의 인코딩을 읽지 못했습니다."
    )


def clean_subtitle_file(text):
    text = str(text)

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

    text = re.sub(r"<[^>]+>", " ", text)

    lines = []
    previous_line = ""

    for line in text.splitlines():
        line = clean_text(line)

        if not line:
            continue

        if line == previous_line:
            continue

        lines.append(line)
        previous_line = line

    return limit_text(" ".join(lines))


def read_uploaded_subtitle(uploaded_file):
    decoded_text = decode_uploaded_file(
        uploaded_file
    )

    subtitle_text = clean_subtitle_file(
        decoded_text
    )

    if len(tokenize_mixed_text(subtitle_text)) < 5:
        raise ValueError(
            "업로드한 자막의 텍스트가 너무 짧습니다."
        )

    return subtitle_text


# =========================================================
# 5. YouTube URL과 API
# =========================================================

def extract_video_id(url):
    if not url:
        return None

    url = str(url).strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)

    hostname = (
        parsed.hostname or ""
    ).lower().replace("www.", "")

    if hostname == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]

        if len(video_id) == 11:
            return video_id

    if hostname in {
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(
                parsed.query
            ).get("v", [None])[0]

            if video_id and len(video_id) == 11:
                return video_id

        path_parts = parsed.path.strip("/").split("/")

        if (
            len(path_parts) >= 2
            and path_parts[0]
            in {"shorts", "embed", "live"}
        ):
            video_id = path_parts[1]

            if len(video_id) == 11:
                return video_id

    match = re.search(
        r"(?:v=|youtu\.be/|shorts/|embed/|live/)"
        r"([A-Za-z0-9_-]{11})",
        url,
    )

    if match:
        return match.group(1)

    return None


def get_api_key():
    try:
        secret_key = st.secrets["YOUTUBE_API_KEY"]

        if secret_key:
            return secret_key

    except Exception:
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

    now = datetime.now(timezone.utc)

    return max(
        (now - published_date).days,
        1,
    )


def get_thumbnail(snippet):
    thumbnails = snippet.get("thumbnails", {})

    if "high" in thumbnails:
        return thumbnails["high"].get("url", "")

    if "medium" in thumbnails:
        return thumbnails["medium"].get("url", "")

    if "default" in thumbnails:
        return thumbnails["default"].get("url", "")

    return ""


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

    published_at = snippet.get(
        "publishedAt",
        "",
    )

    views = int(
        statistics.get("viewCount", 0)
    )

    likes = int(
        statistics.get("likeCount", 0)
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
            + video_id
        ),
    }


# =========================================================
# 6. YouTube 자막
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_transcript(video_id):
    """
    youtube-transcript-api 1.2.3 기준입니다.
    """

    empty_result = {
        "success": False,
        "text": "",
        "language": "none",
        "generated": False,
        "translated": False,
        "error_type": "",
        "error_message": "",
    }

    try:
        api = YouTubeTranscriptApi()

        transcript_list = api.list(video_id)

        transcripts = list(transcript_list)

    except Exception as error:
        result = empty_result.copy()

        result["error_type"] = type(error).__name__
        result["error_message"] = (
            str(error) or repr(error)
        )

        return result

    if not transcripts:
        result = empty_result.copy()

        result["error_type"] = "EmptyTranscriptList"
        result["error_message"] = (
            "사용 가능한 자막 목록이 비어 있습니다."
        )

        return result

    def transcript_priority(transcript):
        language_code = str(
            getattr(
                transcript,
                "language_code",
                "",
            )
        ).lower()

        generated = bool(
            getattr(
                transcript,
                "is_generated",
                False,
            )
        )

        if language_code.startswith("ko"):
            return 1 if generated else 0

        if language_code.startswith("en"):
            return 3 if generated else 2

        return 5 if generated else 4

    transcripts = sorted(
        transcripts,
        key=transcript_priority,
    )

    fetch_errors = []

    for transcript in transcripts:
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

            transcript_text = limit_text(
                " ".join(text_parts)
            )

            if (
                len(
                    tokenize_mixed_text(
                        transcript_text
                    )
                )
                >= 5
            ):
                return {
                    "success": True,
                    "text": transcript_text,
                    "language": str(
                        getattr(
                            transcript,
                            "language_code",
                            "unknown",
                        )
                    ),
                    "generated": bool(
                        getattr(
                            transcript,
                            "is_generated",
                            False,
                        )
                    ),
                    "translated": False,
                    "error_type": "",
                    "error_message": "",
                }

        except Exception as error:
            fetch_errors.append(
                type(error).__name__
                + ": "
                + str(error)
            )

    result = empty_result.copy()

    result["error_type"] = "TranscriptFetchFailed"
    result["error_message"] = "\n".join(
        fetch_errors
    )

    return result


# =========================================================
# 7. 입력 영상 텍스트 준비
# =========================================================

def prepare_source_text(
    video_id,
    video_details,
    manual_transcript,
    uploaded_subtitle,
):
    if manual_transcript.strip():
        text = limit_text(manual_transcript)

        if len(tokenize_mixed_text(text)) < 5:
            raise ValueError(
                "직접 입력한 자막이 너무 짧습니다."
            )

        return text, "직접 입력한 자막"

    if uploaded_subtitle is not None:
        text = read_uploaded_subtitle(
            uploaded_subtitle
        )

        return (
            text,
            "업로드 자막: "
            + uploaded_subtitle.name,
        )

    transcript_result = get_transcript(video_id)

    if transcript_result["success"]:
        if transcript_result["translated"]:
            label = "YouTube 번역 자막"

        elif transcript_result["generated"]:
            label = "YouTube 자동 생성 자막"

        else:
            label = "YouTube 수동 등록 자막"

        return (
            transcript_result["text"],
            label,
        )

    fallback_text = clean_text(
        video_details["title"]
        + " "
        + video_details["title"]
        + " "
        + video_details["description"]
    )

    if len(tokenize_mixed_text(fallback_text)) < 5:
        raise ValueError(
            "자막 수집에 실패했고 제목·설명도 "
            "분석하기에 너무 짧습니다. "
            "자막을 직접 입력하거나 파일을 업로드하세요. "
            "자막 오류: "
            + transcript_result["error_type"]
        )

    return (
        fallback_text,
        "YouTube 자막 수집 실패: 제목·설명 사용",
    )


# =========================================================
# 8. TF-IDF 분석
# =========================================================

def analyze_tfidf(
    source_text,
    sentence_count,
    keyword_count=20,
):
    sentences = split_sentences(source_text)

    if not sentences:
        raise ValueError(
            "분석 가능한 문장을 찾지 못했습니다."
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

    if matrix.shape[1] == 0:
        raise ValueError(
            "TF-IDF 단어 벡터가 비어 있습니다."
        )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    sentence_scores = np.asarray(
        matrix.sum(axis=1)
    ).ravel()

    summary_count = min(
        int(sentence_count),
        len(sentences),
    )

    top_sentence_indices = np.argsort(
        sentence_scores
    )[::-1][:summary_count]

    ordered_summary_indices = sorted(
        top_sentence_indices.tolist()
    )

    summary = " ".join(
        sentences[index]
        for index in ordered_summary_indices
    )

    sentence_rows = []

    for rank, index in enumerate(
        top_sentence_indices,
        start=1,
    ):
        sentence_rows.append(
            {
                "순위": rank,
                "문장 번호": int(index + 1),
                "TF-IDF 점수": round(
                    float(
                        sentence_scores[index]
                    ),
                    6,
                ),
                "문장": sentences[index],
            }
        )

    sentence_score_df = pd.DataFrame(
        sentence_rows
    )

    total_term_scores = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    keyword_indices = np.argsort(
        total_term_scores
    )[::-1][:keyword_count]

    keywords = []

    keyword_rows = []

    for rank, index in enumerate(
        keyword_indices,
        start=1,
    ):
        keyword = str(
            feature_names[index]
        )

        keywords.append(keyword)

        keyword_rows.append(
            {
                "순위": rank,
                "단어 또는 구": keyword,
                "TF-IDF 점수": round(
                    float(
                        total_term_scores[index]
                    ),
                    6,
                ),
            }
        )

    keyword_df = pd.DataFrame(
        keyword_rows
    )

    feature_columns = []

    for index, feature_name in enumerate(
        feature_names
    ):
        safe_name = re.sub(
            r"[^가-힣A-Za-z0-9_]+",
            "_",
            str(feature_name),
        )

        feature_columns.append(
            "tfidf_"
            + str(index).zfill(4)
            + "_"
            + safe_name
        )

    sentence_feature_df = pd.DataFrame(
        matrix.toarray(),
        columns=feature_columns,
    )

    sentence_feature_df.insert(
        0,
        "sentence",
        sentences,
    )

    sentence_feature_df.insert(
        1,
        "sentence_score",
        sentence_scores,
    )

    return {
        "summary": summary,
        "keywords": keywords,
        "keyword_df": keyword_df,
        "sentence_score_df": sentence_score_df,
        "sentence_feature_df": sentence_feature_df,
    }


# =========================================================
# 9. 후보 검색
# =========================================================

def build_search_query(
    video_title,
    keywords,
    custom_query,
):
    if custom_query.strip():
        return clean_text(custom_query)[:100]

    title_tokens = tokenize_mixed_text(
        video_title
    )

    keyword_tokens = []

    for keyword in keywords:
        parts = str(keyword).split()

        for part in parts:
            part = clean_text(part)

            if (
                len(part) >= 2
                and part not in keyword_tokens
            ):
                keyword_tokens.append(part)

    query_tokens = []

    for token in (
        title_tokens[:3]
        + keyword_tokens[:3]
    ):
        if token not in query_tokens:
            query_tokens.append(token)

    return " ".join(query_tokens)[:100]


@st.cache_data(ttl=3600, show_spinner=False)
def search_candidates(
    search_query,
    api_key,
    maximum_results,
):
    youtube = get_youtube_client(api_key)

    search_response = (
        youtube.search()
        .list(
            part="snippet",
            q=search_query,
            type="video",
            maxResults=min(
                int(maximum_results),
                50,
            ),
            order="relevance",
            safeSearch="moderate",
        )
        .execute()
    )

    video_ids = []

    for item in search_response.get(
        "items",
        [],
    ):
        video_id = item.get(
            "id",
            {},
        ).get("videoId")

        if video_id:
            video_ids.append(video_id)

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

    item_by_id = {}

    for item in details_response.get(
        "items",
        [],
    ):
        item_by_id[item.get("id", "")] = item

    candidates = []

    for video_id in video_ids:
        item = item_by_id.get(video_id)

        if item is None:
            continue

        snippet = item.get("snippet", {})
        statistics = item.get(
            "statistics",
            {},
        )

        title = clean_text(
            snippet.get("title", "")
        )

        description = clean_text(
            snippet.get("description", "")
        )

        comparison_text = clean_text(
            title
            + " "
            + title
            + " "
            + title
            + " "
            + description
        )

        if (
            len(
                tokenize_mixed_text(
                    comparison_text
                )
            )
            < 2
        ):
            continue

        published_at = snippet.get(
            "publishedAt",
            "",
        )

        views = int(
            statistics.get("viewCount", 0)
        )

        likes = int(
            statistics.get("likeCount", 0)
        )

        days = calculate_days_since_upload(
            published_at
        )

        candidates.append(
            {
                "video_id": video_id,
                "title": title,
                "description": description,
                "comparison_text": comparison_text,
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
                    "https://www.youtube.com/watch?v="
                    + video_id
                ),
            }
        )

    return candidates


# =========================================================
# 10. 코사인 유사도와 상대 점수
# =========================================================

def calculate_relative_recommendations(
    source_details,
    tfidf_result,
    candidates,
):
    if not candidates:
        raise ValueError(
            "비교할 후보 영상이 없습니다."
        )

    source_comparison_text = clean_text(
        source_details["title"]
        + " "
        + source_details["title"]
        + " "
        + source_details["title"]
        + " "
        + tfidf_result["summary"]
        + " "
        + " ".join(
            tfidf_result["keywords"][:15]
        )
    )

    candidate_texts = [
        candidate["comparison_text"]
        for candidate in candidates
    ]

    all_texts = [
        source_comparison_text
    ] + candidate_texts

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
        all_texts
    )

    if matrix.shape[1] == 0:
        raise ValueError(
            "코사인 유사도를 계산할 단어가 없습니다."
        )

    source_vector = matrix[0:1]
    candidate_vectors = matrix[1:]

    cosine_values = cosine_similarity(
        source_vector,
        candidate_vectors,
    ).ravel()

    if len(cosine_values) == 0:
        raise ValueError(
            "코사인 유사도 결과가 비어 있습니다."
        )

    maximum_similarity = float(
        np.max(cosine_values)
    )

    if maximum_similarity > 0:
        relative_values = (
            cosine_values
            / maximum_similarity
        )
    else:
        relative_values = np.zeros(
            len(cosine_values),
            dtype=float,
        )

    rows = []

    for index, candidate in enumerate(
        candidates
    ):
        row = candidate.copy()

        row["cosine_similarity"] = float(
            cosine_values[index]
        )

        row["relative_similarity"] = float(
            relative_values[index]
        )

        row["relative_percent"] = float(
            relative_values[index] * 100.0
        )

        rows.append(row)

    all_results_df = pd.DataFrame(rows)

    filtered_df = all_results_df[
        (
            all_results_df[
                "cosine_similarity"
            ]
            >= ABSOLUTE_MINIMUM_SIMILARITY
        )
        &
        (
            all_results_df[
                "relative_similarity"
            ]
            >= RELATIVE_MINIMUM_SIMILARITY
        )
    ].copy()

    # 최고값이 0보다 크면 최고 후보는 반드시 포함합니다.
    if (
        filtered_df.empty
        and maximum_similarity > 0
    ):
        best_index = all_results_df[
            "cosine_similarity"
        ].idxmax()

        filtered_df = all_results_df.loc[
            [best_index]
        ].copy()

        filtered_df[
            "best_candidate_only"
        ] = True

    else:
        filtered_df[
            "best_candidate_only"
        ] = False

    filtered_df = filtered_df.sort_values(
        by="cosine_similarity",
        ascending=False,
    ).reset_index(drop=True)

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    feature_columns = []

    for index, feature_name in enumerate(
        feature_names
    ):
        safe_name = re.sub(
            r"[^가-힣A-Za-z0-9_]+",
            "_",
            str(feature_name),
        )

        feature_columns.append(
            "tfidf_"
            + str(index).zfill(4)
            + "_"
            + safe_name
        )

    candidate_feature_df = pd.DataFrame(
        candidate_vectors.toarray(),
        columns=feature_columns,
    )

    metadata_columns = [
        "video_id",
        "title",
        "channel_title",
        "url",
        "views",
        "likes",
        "daily_views",
        "cosine_similarity",
        "relative_similarity",
        "relative_percent",
    ]

    metadata_df = all_results_df[
        metadata_columns
    ].reset_index(drop=True)

    orange_df = pd.concat(
        [
            metadata_df,
            candidate_feature_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    candidate_lookup = {}

    for candidate in candidates:
        candidate_lookup[
            candidate["video_id"]
        ] = candidate

    return {
        "recommendations": filtered_df,
        "all_results": all_results_df,
        "orange_df": orange_df,
        "maximum_similarity": maximum_similarity,
        "candidate_lookup": candidate_lookup,
    }


def create_pair_tfidf_table(
    source_details,
    tfidf_result,
    candidate,
):
    source_text = clean_text(
        source_details["title"]
        + " "
        + source_details["title"]
        + " "
        + source_details["title"]
        + " "
        + tfidf_result["summary"]
        + " "
        + " ".join(
            tfidf_result["keywords"][:15]
        )
    )

    candidate_text = candidate[
        "comparison_text"
    ]

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
        [
            source_text,
            candidate_text,
        ]
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    idf_values = vectorizer.idf_

    source_values = (
        matrix[0].toarray().ravel()
    )

    candidate_values = (
        matrix[1].toarray().ravel()
    )

    detail_df = pd.DataFrame(
        {
            "단어": feature_names,
            "IDF": idf_values,
            "입력 영상 TF-IDF": source_values,
            "후보 영상 TF-IDF": candidate_values,
            "TF-IDF 합계": (
                source_values
                + candidate_values
            ),
        }
    )

    return detail_df.sort_values(
        by="TF-IDF 합계",
        ascending=False,
    ).head(30)


def to_csv_bytes(dataframe):
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# =========================================================
# 11. 세션 상태
# =========================================================

SESSION_DEFAULTS = {
    "source_details": None,
    "source_text": "",
    "source_label": "",
    "tfidf_result": None,
    "recommendation_result": None,
    "search_query": "",
}

for session_key, default_value in (
    SESSION_DEFAULTS.items()
):
    if session_key not in st.session_state:
        st.session_state[
            session_key
        ] = default_value


def reset_session_results():
    st.session_state.source_details = None
    st.session_state.source_text = ""
    st.session_state.source_label = ""
    st.session_state.tfidf_result = None
    st.session_state.recommendation_result = None
    st.session_state.search_query = ""


def perform_tfidf_analysis(
    youtube_url,
    api_key,
    manual_transcript,
    uploaded_subtitle,
    sentence_count,
):
    video_id = extract_video_id(
        youtube_url
    )

    if not video_id:
        raise ValueError(
            "올바른 YouTube URL을 입력하세요."
        )

    video_details = get_video_details(
        video_id,
        api_key,
    )

    if video_details is None:
        raise ValueError(
            "영상 정보를 찾지 못했습니다."
        )

    source_text, source_label = (
        prepare_source_text(
            video_id=video_id,
            video_details=video_details,
            manual_transcript=(
                manual_transcript
            ),
            uploaded_subtitle=(
                uploaded_subtitle
            ),
        )
    )

    tfidf_result = analyze_tfidf(
        source_text=source_text,
        sentence_count=sentence_count,
    )

    st.session_state.source_details = (
        video_details
    )

    st.session_state.source_text = (
        source_text
    )

    st.session_state.source_label = (
        source_label
    )

    st.session_state.tfidf_result = (
        tfidf_result
    )

    return (
        video_id,
        video_details,
        tfidf_result,
    )


# =========================================================
# 12. 화면
# =========================================================

st.title(
    "🎬 YouTube TF-IDF 분석 및 상대 코사인 추천"
)

st.write(
    "TF-IDF 자막 분석과 코사인 유사도 추천을 "
    "서로 분리하여 실행합니다."
)

with st.expander(
    "분석 원리"
):
    st.markdown(
        r"""
### TF-IDF 분석

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

입력 영상의 자막에서 핵심 문장과 핵심 키워드를
추출합니다.

### 원본 코사인 유사도

\[
CosineSimilarity(A,B)
=
\frac{A\cdot B}
{\|A\|\|B\|}
\]

### 상대 유사도

\[
RelativeSimilarity_i
=
\frac{CosineSimilarity_i}
{\max(CosineSimilarity)}
\]

후보 중 가장 높은 영상이 100%가 되고,
최고 후보의 30% 이상인 영상들을 상대적으로 추천합니다.
        """
    )

if st.sidebar.button(
    "🗑️ 캐시 및 결과 초기화",
    use_container_width=True,
):
    st.cache_data.clear()
    reset_session_results()

    st.sidebar.success(
        "초기화했습니다."
    )

with st.sidebar:
    st.header("설정")

    summary_sentence_count = st.slider(
        "요약 문장 수",
        min_value=3,
        max_value=10,
        value=5,
    )

    candidate_count = st.slider(
        "후보 영상 수",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    recommendation_count = st.slider(
        "표시할 추천 수",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.info(
        "추천 기준\n\n"
        "- 절대 유사도: 0.001 이상\n"
        "- 최고 후보 대비 30% 이상\n"
        "- 최고 후보는 항상 표시"
    )

api_key = get_api_key()

if not api_key:
    st.warning(
        "Streamlit Secrets에 "
        "YOUTUBE_API_KEY를 설정하세요."
    )

youtube_url = st.text_input(
    "YouTube 영상 URL",
    placeholder=(
        "https://www.youtube.com/"
        "watch?v=..."
    ),
)

custom_search_query = st.text_input(
    "후보 영상 검색어(선택)",
    placeholder=(
        "예: 인공지능 교육 활용"
    ),
)

with st.expander(
    "자막 직접 입력 또는 업로드"
):
    manual_transcript = st.text_area(
        "자막 직접 입력",
        height=150,
    )

    uploaded_subtitle = st.file_uploader(
        "TXT, SRT, VTT 자막 파일",
        type=[
            "txt",
            "srt",
            "vtt",
        ],
    )

button_column_1, button_column_2, button_column_3 = (
    st.columns(3)
)

with button_column_1:
    diagnose_button = st.button(
        "🔍 자막 진단",
        use_container_width=True,
    )

with button_column_2:
    tfidf_button = st.button(
        "📊 TF-IDF 분석",
        type="primary",
        use_container_width=True,
    )

with button_column_3:
    recommendation_button = st.button(
        "🎯 상대 유사도 추천",
        use_container_width=True,
    )


# =========================================================
# 13. 자막 진단
# =========================================================

if diagnose_button:
    diagnosis_video_id = extract_video_id(
        youtube_url
    )

    if not diagnosis_video_id:
        st.error(
            "올바른 YouTube URL을 입력하세요."
        )

    else:
        with st.spinner(
            "자막을 진단하고 있습니다..."
        ):
            diagnosis = get_transcript(
                diagnosis_video_id
            )

        if diagnosis["success"]:
            st.success(
                "자막 수집 성공"
            )

            metric_1, metric_2 = st.columns(2)

            metric_1.metric(
                "언어",
                diagnosis["language"],
            )

            metric_2.metric(
                "자막 유형",
                (
                    "자동 생성"
                    if diagnosis["generated"]
                    else "수동 등록"
                ),
            )

            st.text_area(
                "자막 미리보기",
                diagnosis["text"][:2000],
                height=250,
            )

        else:
            st.error(
                "자막 자동 수집 실패"
            )

            st.write(
                "오류 유형: "
                + diagnosis["error_type"]
            )

            st.code(
                diagnosis["error_message"]
                or "오류 메시지가 없습니다."
            )


# =========================================================
# 14. TF-IDF 버튼
# =========================================================

if tfidf_button:
    if not api_key:
        st.error(
            "API 키를 먼저 설정하세요."
        )

    else:
        try:
            with st.spinner(
                "TF-IDF 분석 중입니다..."
            ):
                perform_tfidf_analysis(
                    youtube_url=(
                        youtube_url
                    ),
                    api_key=api_key,
                    manual_transcript=(
                        manual_transcript
                    ),
                    uploaded_subtitle=(
                        uploaded_subtitle
                    ),
                    sentence_count=(
                        summary_sentence_count
                    ),
                )

            st.session_state[
                "recommendation_result"
            ] = None

            st.session_state[
                "search_query"
            ] = ""

            st.success(
                "TF-IDF 분석 완료"
            )

        except Exception as error:
            st.error(
                "TF-IDF 분석 실패"
            )

            st.write(
                "오류 유형: "
                + type(error).__name__
            )

            st.write(
                "오류 내용: "
                + str(error)
            )

            with st.expander(
                "상세 오류"
            ):
                st.code(
                    traceback.format_exc()
                )


# =========================================================
# 15. 추천 버튼
# =========================================================

if recommendation_button:
    if not api_key:
        st.error(
            "API 키를 먼저 설정하세요."
        )

    else:
        try:
            with st.spinner(
                "입력 영상 분석 중입니다..."
            ):
                (
                    source_video_id,
                    source_details,
                    tfidf_result,
                ) = perform_tfidf_analysis(
                    youtube_url=(
                        youtube_url
                    ),
                    api_key=api_key,
                    manual_transcript=(
                        manual_transcript
                    ),
                    uploaded_subtitle=(
                        uploaded_subtitle
                    ),
                    sentence_count=(
                        summary_sentence_count
                    ),
                )

            search_query = build_search_query(
                video_title=(
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
                "후보 검색 및 유사도 계산 중입니다..."
            ):
                candidates = search_candidates(
                    search_query=(
                        search_query
                    ),
                    api_key=api_key,
                    maximum_results=(
                        candidate_count
                    ),
                )

                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["video_id"]
                    != source_video_id
                ]

                recommendation_result = (
                    calculate_relative_recommendations(
                        source_details=(
                            source_details
                        ),
                        tfidf_result=(
                            tfidf_result
                        ),
                        candidates=(
                            candidates
                        ),
                    )
                )

            st.session_state[
                "recommendation_result"
            ] = recommendation_result

            st.session_state[
                "search_query"
            ] = search_query

            st.success(
                "상대 유사도 추천 완료"
            )

        except Exception as error:
            st.error(
                "추천 실행 실패"
            )

            st.write(
                "오류 유형: "
                + type(error).__name__
            )

            st.write(
                "오류 내용: "
                + str(error)
            )

            with st.expander(
                "상세 오류"
            ):
                st.code(
                    traceback.format_exc()
                )


# =========================================================
# 16. 입력 영상 정보
# =========================================================

if (
    st.session_state.source_details
    is not None
):
    source_details = (
        st.session_state.source_details
    )

    st.divider()

    st.subheader("입력 영상")

    video_column, info_column = (
        st.columns([1, 2])
    )

    with video_column:
        st.video(
            source_details["url"]
        )

    with info_column:
        st.markdown(
            "### "
            + source_details["title"]
        )

        st.write(
            "채널: "
            + source_details[
                "channel_title"
            ]
        )

        st.write(
            "분석 데이터: "
            + st.session_state[
                "source_label"
            ]
        )

        st.write(
            "조회수: "
            + format(
                source_details["views"],
                ",",
            )
        )

        st.write(
            "좋아요: "
            + format(
                source_details["likes"],
                ",",
            )
        )


# =========================================================
# 17. TF-IDF 결과
# =========================================================

if (
    st.session_state.tfidf_result
    is not None
):
    tfidf_result = (
        st.session_state.tfidf_result
    )

    st.divider()

    st.header(
        "📊 TF-IDF 분석 결과"
    )

    st.subheader(
        "핵심 내용 요약"
    )

    st.write(
        tfidf_result["summary"]
    )

    st.subheader(
        "핵심 키워드"
    )

    st.dataframe(
        tfidf_result["keyword_df"],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "문장별 TF-IDF 점수"
    ):
        st.dataframe(
            tfidf_result[
                "sentence_score_df"
            ],
            use_container_width=True,
            hide_index=True,
        )

    tfidf_download_1, tfidf_download_2 = (
        st.columns(2)
    )

    with tfidf_download_1:
        st.download_button(
            "키워드 CSV",
            data=to_csv_bytes(
                tfidf_result[
                    "keyword_df"
                ]
            ),
            file_name=(
                "tfidf_keywords.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with tfidf_download_2:
        st.download_button(
            "문장 벡터 CSV",
            data=to_csv_bytes(
                tfidf_result[
                    "sentence_feature_df"
                ]
            ),
            file_name=(
                "tfidf_sentence_vectors.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# 18. 추천 결과
# =========================================================

recommendation_result = (
    st.session_state.recommendation_result
)

if recommendation_result is not None:
    recommendations = (
        recommendation_result[
            "recommendations"
        ]
    )

    st.divider()

    st.header(
        "🎯 코사인 유사도 추천"
    )

    st.write(
        "사용한 검색어: `"
        + st.session_state[
            "search_query"
        ]
        + "`"
    )

    maximum_similarity = (
        recommendation_result[
            "maximum_similarity"
        ]
    )

    summary_metric_1, summary_metric_2 = (
        st.columns(2)
    )

    summary_metric_1.metric(
        "최고 원본 코사인 유사도",
        format(
            maximum_similarity,
            ".4f",
        ),
    )

    summary_metric_2.metric(
        "상대 추천 기준",
        format(
            RELATIVE_MINIMUM_SIMILARITY
            * 100,
            ".0f",
        )
        + "%",
    )

    if (
        maximum_similarity
        < LOW_QUALITY_WARNING
    ):
        st.warning(
            "후보 전체의 원본 유사도가 낮습니다. "
            "상대 점수는 후보들 사이의 순위이지, "
            "절대적으로 관련성이 높다는 뜻은 아닙니다."
        )

    if recommendations.empty:
        st.warning(
            "코사인 유사도가 0보다 큰 "
            "후보를 찾지 못했습니다."
        )

    else:
        shown_recommendations = (
            recommendations.head(
                recommendation_count
            )
        )

        for rank, row_tuple in enumerate(
            shown_recommendations.itertuples(
                index=False
            ),
            start=1,
        ):
            row = row_tuple._asdict()

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
                    "### "
                    + str(rank)
                    + ". ["
                    + row["title"]
                    + "]("
                    + row["url"]
                    + ")"
                )

                st.write(
                    "채널: "
                    + row["channel_title"]
                )

                score_column_1, score_column_2 = (
                    st.columns(2)
                )

                score_column_1.metric(
                    "원본 코사인 유사도",
                    format(
                        row[
                            "cosine_similarity"
                        ],
                        ".4f",
                    ),
                )

                score_column_2.metric(
                    "상대 유사도",
                    format(
                        row[
                            "relative_percent"
                        ],
                        ".1f",
                    )
                    + "%",
                )

                progress_value = float(
                    row[
                        "relative_similarity"
                    ]
                )

                progress_value = max(
                    0.0,
                    min(
                        progress_value,
                        1.0,
                    ),
                )

                st.progress(
                    progress_value
                )

                st.write(
                    "조회수: "
                    + format(
                        int(row["views"]),
                        ",",
                    )
                    + " | 좋아요: "
                    + format(
                        int(row["likes"]),
                        ",",
                    )
                )

        st.subheader(
            "상대 유사도 비교"
        )

        relative_chart = (
            shown_recommendations[
                [
                    "title",
                    "relative_similarity",
                ]
            ]
            .set_index("title")
        )

        st.bar_chart(
            relative_chart
        )

        st.subheader(
            "원본 코사인 유사도 비교"
        )

        cosine_chart = (
            shown_recommendations[
                [
                    "title",
                    "cosine_similarity",
                ]
            ]
            .set_index("title")
        )

        st.bar_chart(
            cosine_chart
        )

        first_video_id = str(
            shown_recommendations.iloc[
                0
            ]["video_id"]
        )

        candidate_lookup = (
            recommendation_result[
                "candidate_lookup"
            ]
        )

        first_candidate = (
            candidate_lookup[
                first_video_id
            ]
        )

        pair_tfidf_df = (
            create_pair_tfidf_table(
                source_details=(
                    st.session_state[
                        "source_details"
                    ]
                ),
                tfidf_result=(
                    st.session_state[
                        "tfidf_result"
                    ]
                ),
                candidate=(
                    first_candidate
                ),
            )
        )

        st.subheader(
            "추천 1위와 TF-IDF 비교"
        )

        st.dataframe(
            pair_tfidf_df,
            use_container_width=True,
            hide_index=True,
        )

        download_1, download_2, download_3 = (
            st.columns(3)
        )

        with download_1:
            st.download_button(
                "추천 결과 CSV",
                data=to_csv_bytes(
                    recommendations
                ),
                file_name=(
                    "relative_recommendations.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with download_2:
            st.download_button(
                "전체 후보 CSV",
                data=to_csv_bytes(
                    recommendation_result[
                        "all_results"
                    ]
                ),
                file_name=(
                    "all_candidate_scores.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with download_3:
            st.download_button(
                "Orange3 TF-IDF CSV",
                data=to_csv_bytes(
                    recommendation_result[
                        "orange_df"
                    ]
                ),
                file_name=(
                    "orange_candidate_tfidf.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )
