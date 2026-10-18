import os
import re
import traceback
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from groq import Groq
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
)
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="YouTube TF-IDF 및 유사도 분석",
    page_icon="🎬",
    layout="wide",
)

MAX_TRANSCRIPT_CHARACTERS = 30000
MAX_TFIDF_FEATURES = 1500

ABSOLUTE_MINIMUM_SIMILARITY = 0.001
RELATIVE_MINIMUM_SIMILARITY = 0.30
LOW_QUALITY_WARNING = 0.03

SIMILARITY_METHODS = {
    "cosine": "코사인 유사도",
    "euclidean": "유클리드 유사도",
    "jaccard": "자카드 유사도",
}   # "combined" 삭제

# [신규] 데이터별 권장 가이드 (덩어리 3의 가이드 탭에서 표시)
METHOD_GUIDE = {
    "cosine": {
        "best_for": "자막·긴 설명처럼 길거나, 비교 텍스트끼리 길이 차이가 큰 데이터",
        "why": "단어 중요도(TF-IDF)를 반영하고 벡터 방향만 비교해서 길이 차이에 둔감합니다.",
        "weak": "공통 단어가 거의 없는 아주 짧은 텍스트에서는 0에 가깝게 나옵니다.",
    },
    "euclidean": {
        "best_for": "순위용이 아니라 점수 해석용 (0~1 변환 점수가 필요할 때)",
        "why": "L2 정규화된 TF-IDF에서는 코사인과 순위가 수학적으로 같습니다.",
        "weak": "코사인 외에 추가되는 정보가 없습니다.",
    },
    "jaccard": {
        "best_for": "제목·키워드 목록처럼 양쪽이 모두 짧은 데이터",
        "why": "단어 빈도·IDF에 의존하지 않고 겹치는 단어 비율만 봅니다. 후보 수가 적어 IDF가 불안정할 때도 안정적입니다.",
        "weak": "한쪽이 훨씬 길면 합집합이 커져 점수가 눌립니다. 단어 중요도도 반영하지 못합니다.",
    },
}

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"


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


def get_groq_api_key():
    try:
        secret_key = st.secrets["GROQ_API_KEY"]

        if secret_key:
            return secret_key

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "")


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

    extractive_summary = " ".join(
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
        "extractive_summary": extractive_summary,
        "keywords": keywords,
        "keyword_df": keyword_df,
        "sentence_score_df": sentence_score_df,
        "sentence_feature_df": sentence_feature_df,
    }


# =========================================================
# 8-1. Groq 기반 자연어 요약
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_summary(
    video_title,
    extractive_summary,
    keywords,
    groq_api_key,
):
    """
    TF-IDF로 추출한 핵심 문장과 키워드를 Groq(LLM)에게 전달하여
    자연스러운 문장으로 재구성한 요약을 생성합니다.

    실패하거나 API 키가 없으면 success=False를 반환하며,
    호출부에서 기존 추출 요약으로 폴백 처리합니다.
    """

    empty_result = {
        "success": False,
        "text": "",
        "error_type": "",
        "error_message": "",
    }

    if not groq_api_key:
        result = empty_result.copy()
        result["error_type"] = "NoApiKey"
        result["error_message"] = (
            "Groq API 키가 설정되지 않았습니다."
        )
        return result

    if not extractive_summary.strip():
        result = empty_result.copy()
        result["error_type"] = "EmptyInput"
        result["error_message"] = (
            "요약할 원본 문장이 없습니다."
        )
        return result

    try:
        client = Groq(api_key=groq_api_key)

        keyword_text = ", ".join(keywords[:15])

        system_prompt = (
            "너는 유튜브 영상 내용을 사람들에게 "
            "친근하고 이해하기 쉽게 설명해주는 "
            "어시스턴트야. 항상 자연스러운 "
            "한국어 존댓말로 답변해."
        )

        user_prompt = (
            "아래는 영상 제목과, 자막에서 자동으로 "
            "추출한 핵심 문장들, 핵심 키워드야. "
            "이 정보를 바탕으로 영상이 어떤 내용을 "
            "다루는지 3~5문장 정도의 자연스러운 "
            "설명글로 다시 써줘.\n\n"
            "규칙:\n"
            "- 추출된 문장을 그대로 복사하지 말고, "
            "내용을 이해한 뒤 너의 말로 재구성해줘.\n"
            "- 실제 영상에 없는 내용을 지어내지 마.\n"
            "- 마크다운 기호(#, *, - 등)는 쓰지 마.\n\n"
            "영상 제목: " + video_title + "\n\n"
            "추출된 핵심 문장들:\n"
            + extractive_summary + "\n\n"
            "핵심 키워드: " + keyword_text
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.4,
            max_tokens=500,
        )

        summary_text = clean_text(
            response.choices[0].message.content
        )

        if not summary_text:
            result = empty_result.copy()
            result["error_type"] = "EmptyResponse"
            result["error_message"] = (
                "Groq 응답이 비어 있습니다."
            )
            return result

        return {
            "success": True,
            "text": summary_text,
            "error_type": "",
            "error_message": "",
        }

    except Exception as error:
        result = empty_result.copy()
        result["error_type"] = type(error).__name__
        result["error_message"] = (
            str(error) or repr(error)
        )
        return result

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
# 10. 코사인 / 유클리드 / 자카드 유사도
# =========================================================
def method_columns(key):
    return (f"{key}_similarity", f"{key}_relative", f"{key}_relative_percent")


def build_source_comparison_text(source_details, tfidf_result):
    return clean_text(
        " ".join([source_details["title"]] * 3)
        + " " + tfidf_result["extractive_summary"]
        + " " + " ".join(tfidf_result["keywords"][:15])
    )


def recommend_similarity_method(source_text, candidate_texts):
    """이번 데이터의 길이 특성을 보고 가장 적합한 유사도와 이유를 반환합니다."""
    source_n = len(tokenize_mixed_text(source_text))
    cand_ns = [len(tokenize_mixed_text(t)) for t in candidate_texts] or [0]
    median_n = float(np.median(cand_ns))
    ratio = median_n / source_n if source_n else 0.0

    if source_n >= 150 and median_n >= 150:
        best = "cosine"
        reason = (
            f"입력({source_n}토큰)과 후보(중앙값 {median_n:.0f}토큰)가 모두 긴 텍스트라 "
            "단어 중요도를 반영하는 코사인이 가장 적합합니다."
        )
    elif ratio < 0.4 or ratio > 2.5:
        best = "cosine"
        reason = (
            f"입력 {source_n}토큰, 후보 중앙값 {median_n:.0f}토큰으로 길이 차이가 큽니다. "
            "자카드는 합집합이 커져 점수가 눌리므로 길이에 둔감한 코사인을 권장합니다."
        )
    elif median_n < 30 and source_n < 60:
        best = "jaccard"
        reason = (
            f"입력 {source_n}토큰, 후보 중앙값 {median_n:.0f}토큰으로 모두 짧습니다. "
            "짧은 글은 TF-IDF 가중치가 불안정하므로 겹치는 단어 비율(자카드)을 우선하고 "
            "코사인으로 교차 확인하세요."
        )
    else:
        best = "cosine"
        reason = (
            f"입력 {source_n}토큰, 후보 중앙값 {median_n:.0f}토큰으로 길이가 비슷한 중간 길이 텍스트입니다. "
            "핵심어 가중치를 반영하는 코사인을 기본으로 권장합니다."
        )

    return {
        "best": best,
        "reason": reason,
        "source_tokens": source_n,
        "candidate_median_tokens": median_n,
    }


def calculate_relative_recommendations(source_details, tfidf_result, candidates):
    if not candidates:
        raise ValueError("비교할 후보 영상이 없습니다.")

    source_text = build_source_comparison_text(source_details, tfidf_result)
    candidate_texts = [c["comparison_text"] for c in candidates]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=MAX_TFIDF_FEATURES,
    )
    matrix = vectorizer.fit_transform([source_text] + candidate_texts)
    if matrix.shape[1] == 0:
        raise ValueError("유사도를 계산할 단어가 없습니다.")

    source_vector, candidate_vectors = matrix[0:1], matrix[1:]

    values_map = {
        "cosine": cosine_similarity(source_vector, candidate_vectors).ravel(),
        "euclidean": convert_distance_to_similarity(
            euclidean_distances(source_vector, candidate_vectors).ravel()
        ),
        "jaccard": calculate_jaccard_similarity(source_text, candidate_texts),
    }

    all_results_df = pd.DataFrame(candidates)
    similarity_column_map = {}
    maximum_similarity_map = {}

    for key, values in values_map.items():
        absolute_col, relative_col, percent_col = method_columns(key)
        maximum = float(np.max(values))
        relative = values / maximum if maximum > 0 else np.zeros(len(values))

        all_results_df[absolute_col] = values.astype(float)
        all_results_df[relative_col] = relative.astype(float)
        all_results_df[percent_col] = (relative * 100.0).astype(float)

        similarity_column_map[key] = (absolute_col, relative_col, percent_col)
        maximum_similarity_map[key] = maximum

    def filter_for_method(key):
        absolute_col, relative_col, _ = similarity_column_map[key]
        filtered = all_results_df[
            (all_results_df[absolute_col] >= ABSOLUTE_MINIMUM_SIMILARITY)
            & (all_results_df[relative_col] >= RELATIVE_MINIMUM_SIMILARITY)
        ].copy()

        if filtered.empty and maximum_similarity_map[key] > 0:
            best_index = all_results_df[absolute_col].idxmax()
            filtered = all_results_df.loc[[best_index]].copy()
            filtered["best_candidate_only"] = True
        else:
            filtered["best_candidate_only"] = False

        return filtered.sort_values(absolute_col, ascending=False).reset_index(drop=True)

    recommendations_by_method = {k: filter_for_method(k) for k in SIMILARITY_METHODS}

    feature_columns = [
        "tfidf_" + str(i).zfill(4) + "_" + re.sub(r"[^가-힣A-Za-z0-9_]+", "_", str(name))
        for i, name in enumerate(vectorizer.get_feature_names_out())
    ]
    candidate_feature_df = pd.DataFrame(candidate_vectors.toarray(), columns=feature_columns)

    metadata_columns = ["video_id", "title", "channel_title", "url", "views", "likes", "daily_views"]
    for key in SIMILARITY_METHODS:
        metadata_columns += list(similarity_column_map[key])

    orange_df = pd.concat(
        [all_results_df[metadata_columns].reset_index(drop=True), candidate_feature_df],
        axis=1,
    )

    return {
        "recommendations_by_method": recommendations_by_method,
        "similarity_column_map": similarity_column_map,
        "maximum_similarity_map": maximum_similarity_map,
        "all_results": all_results_df,
        "orange_df": orange_df,
        "candidate_lookup": {c["video_id"]: c for c in candidates},
        "source_comparison_text": source_text,
        "method_advice": recommend_similarity_method(source_text, candidate_texts),
    }


def compare_similarity_methods(recommendation_result, top_k=5):
    """유사도별 상위 후보 일치도를 비교합니다. (정답 데이터가 없으므로 참고용)"""
    method_map = recommendation_result["similarity_column_map"]
    all_results = recommendation_result["all_results"]
    keys = list(method_map.keys())

    top_ids = {}
    for key in keys:
        col = method_map[key][0]
        top_ids[key] = (
            all_results.sort_values(col, ascending=False).head(top_k)["video_id"].tolist()
        )

    rows = []
    for key in keys:
        col = method_map[key][0]
        ranked = all_results.sort_values(col, ascending=False)

        overlaps = []
        for other in keys:
            if other == key:
                continue
            a, b = set(top_ids[key]), set(top_ids[other])
            overlaps.append(len(a & b) / len(a | b) if (a | b) else 0.0)

        rows.append({
            "유사도 방식": SIMILARITY_METHODS[key],
            "1위 영상": ranked.iloc[0]["title"] if len(ranked) else "",
            f"상위 {top_k}개 평균 점수": round(float(ranked[col].head(top_k).mean()), 4),
            "다른 방식과의 상위권 일치도": round(float(np.mean(overlaps)), 3),
        })

    return pd.DataFrame(rows), top_ids


def explain_recommendation(
    source_details,
    tfidf_result,
    candidate,
    recommendation_row,
    method_key,
    recommendation_result,
):
    """공통 키워드, 점수, 순위, 데이터 특성을 묶어 추천 이유를 만듭니다."""
    source_text = recommendation_result["source_comparison_text"]
    source_tokens = set(tokenize_mixed_text(source_text))
    candidate_tokens = set(tokenize_mixed_text(candidate["comparison_text"]))
    common = source_tokens & candidate_tokens

    prioritized = [w for w in tfidf_result["keywords"] if w in common]
    if not prioritized:
        prioritized = sorted(common, key=len, reverse=True)

    scores = {k: recommendation_row[f"{k}_similarity"] for k in SIMILARITY_METHODS}

    all_results = recommendation_result["all_results"]
    rank_positions = {}
    for key, (col, _, _) in recommendation_result["similarity_column_map"].items():
        ordered = all_results.sort_values(col, ascending=False).reset_index(drop=True)
        match = ordered.index[ordered["video_id"] == candidate["video_id"]].tolist()
        rank_positions[key] = match[0] + 1 if match else None

    advice = recommendation_result["method_advice"]
    relative_pct = recommendation_row[method_columns(method_key)[2]]

    reasons = []
    if prioritized:
        reasons.append(
            f"입력 영상의 핵심어 {len(prioritized)}개가 이 영상의 제목·설명에도 나옵니다."
        )
    else:
        reasons.append("공통 핵심어는 적지만 전체 단어 분포가 일부 비슷합니다.")

    reasons.append(
        f"{SIMILARITY_METHODS[method_key]} 점수 {scores[method_key]:.4f}로, "
        f"후보 중 1위 대비 {relative_pct:.0f}% 수준입니다."
    )

    ranks = [r for r in rank_positions.values() if r is not None]
    if ranks and max(ranks) <= 3:
        reasons.append("모든 유사도 방식에서 상위 3위 안에 들어 방식과 무관하게 일관된 추천입니다.")
    elif ranks and max(ranks) - min(ranks) >= 5:
        reasons.append("방식에 따라 순위 차이가 커서 한 가지 기준에만 의존한 추천일 수 있습니다.")

    if method_key != advice["best"]:
        reasons.append(
            f"이번 데이터에는 {SIMILARITY_METHODS[advice['best']]}가 권장 방식입니다."
        )

    if candidate.get("daily_views", 0) > source_details.get("daily_views", 0):
        reasons.append(
            f"하루 평균 조회수 {candidate['daily_views']:,.0f}회로 입력 영상보다 활발히 시청되고 있습니다."
        )
    if candidate.get("channel_title") == source_details.get("channel_title"):
        reasons.append("입력 영상과 같은 채널입니다.")

    return {
        "common_keywords": prioritized[:8],
        "scores": scores,
        "rank_positions": rank_positions,
        "reasons": reasons,
    }


def generate_ai_summary(video_title, extractive_summary, keywords, groq_api_key):
    """이미 정의돼 있다면 기존 것을 쓰세요."""
    result = {"success": False, "text": "", "error_type": "", "error_message": ""}

    if not groq_api_key:
        result["error_type"] = "NoGroqKey"
        result["error_message"] = "GROQ_API_KEY가 설정되지 않았습니다."
        return result

    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "너는 유튜브 영상 요약 도우미다. 한국어로 답한다."},
                {
                    "role": "user",
                    "content": (
                        f"제목: {video_title}\n\n"
                        f"핵심 문장:\n{extractive_summary}\n\n"
                        f"핵심 키워드: {', '.join(keywords[:15])}\n\n"
                        "위 내용만 근거로 문장을 복사하지 말고 3~5문장으로 요약하고, "
                        "그 아래에 '주요 포인트'를 불릿 5개 이내로 정리해줘."
                    ),
                },
            ],
        )
        result["text"] = response.choices[0].message.content
        result["success"] = True
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)

    return result
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
    groq_api_key,
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

    ai_summary_result = generate_ai_summary(
        video_title=video_details["title"],
        extractive_summary=(
            tfidf_result["extractive_summary"]
        ),
        keywords=tfidf_result["keywords"],
        groq_api_key=groq_api_key,
    )

    tfidf_result["ai_summary_result"] = (
        ai_summary_result
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
# 12. 화면 상단
# =========================================================

st.title(
    "🎬 YouTube TF-IDF 분석 및 유사도 추천"
)

st.write(
    "TF-IDF 자막 분석과 코사인·유클리드·자카드·통합 "
    "유사도 추천을 서로 분리하여 실행합니다."
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

---

### AI 요약 (Groq / Llama)

TF-IDF로 뽑아낸 핵심 문장과 키워드를 대규모 언어모델에게
전달하여, 문장을 그대로 복사하지 않고 내용을 이해한 뒤
자연스러운 설명으로 다시 작성합니다.

---

### 1) 코사인 유사도 (Cosine Similarity)

\[
CosineSimilarity(A,B)
=
\frac{A\cdot B}
{\|A\|\|B\|}
\]

두 TF-IDF 벡터가 이루는 각도를 기준으로 유사도를 측정합니다.
벡터의 **방향**만 비교하기 때문에 문서 길이 차이에
상대적으로 덜 민감합니다.

---

### 2) 유클리드 유사도 (Euclidean Similarity)

\[
EuclideanDistance(A,B)
=
\sqrt{\sum_i (A_i - B_i)^2}
\]

\[
EuclideanSimilarity(A,B)
=
\frac{1}{1+EuclideanDistance(A,B)}
\]

벡터의 **방향뿐 아니라 크기(길이)**까지 반영하므로,
문서의 길이나 단어 빈도 차이에 코사인 유사도보다
더 민감하게 반응합니다.

---

### 3) 자카드 유사도 (Jaccard Similarity)

\[
JaccardSimilarity(A,B)
=
\frac{|A \cap B|}{|A \cup B|}
\]

TF-IDF 점수를 사용하지 않고, 단어(토큰)의 **존재 여부**만
집합으로 비교합니다.

---

### 4) 통합 유사도 (Combined Similarity)

\[
CombinedSimilarity
=
\frac{1}{3}\Big(
CosineSimilarity
+EuclideanSimilarity
+JaccardSimilarity
\Big)
\]

세 가지 유사도를 각각 33.3%씩 동일한 비중으로 반영하여
평균을 낸 값입니다.

---

### 상대 유사도 (공통)

\[
RelativeSimilarity_i
=
\frac{Similarity_i}
{\max(Similarity)}
\]

선택한 유사도 방식을 기준으로, 후보 중 가장 높은 영상이
100%가 되고, 최고 후보의 30% 이상인 영상들을 상대적으로
추천합니다.
        """
    )

if st.sidebar.button("설정 적용"):          # 콜론(:) 필수! 버튼 라벨 인자 필요
    selected_similarity_method = st.radio(  # if 블록 안이라 4칸 들여쓰기
        "추천 유사도 방식",
        options=list(SIMILARITY_METHODS.keys()),
        format_func=lambda key: SIMILARITY_METHODS[key],
        index=0,  # 코사인 기본값
        help=(
            "유사도 비교 탭에서 상위 후보의 일치도와 "
            "각 방식의 순위를 확인한 뒤 선택하세요."
        ),
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

    st.divider()

    selected_similarity_method = st.radio(
        "유사도 계산 방식 선택",
        options=list(SIMILARITY_METHODS.keys()),
        format_func=lambda key: SIMILARITY_METHODS[key],
        index=0,
    )

    st.info(
        "추천 기준\n\n"
        "- 절대 유사도: 0.001 이상\n"
        "- 최고 후보 대비 30% 이상\n"
        "- 최고 후보는 항상 표시"
    )

api_key = get_api_key()
groq_api_key = get_groq_api_key()

if not api_key:
    st.warning(
        "Streamlit Secrets에 "
        "YOUTUBE_API_KEY를 설정하세요."
    )

if not groq_api_key:
    st.info(
        "GROQ_API_KEY가 설정되지 않아 "
        "AI 자연어 요약 대신 TF-IDF 추출 요약을 "
        "사용합니다."
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
    summary_button = st.button(
        "🤖 영상 요약",
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
        "🎯 유사도 추천",
        use_container_width=True,
    )


# =========================================================
# 13. 영상 요약 (Groq AI)
# =========================================================

if summary_button:
    if not api_key:
        st.error(
            "YouTube API 키를 먼저 설정하세요."
        )

    else:
        summary_video_id = extract_video_id(
            youtube_url
        )

        if not summary_video_id:
            st.error(
                "올바른 YouTube URL을 입력하세요."
            )

        else:
            try:
                with st.spinner(
                    "영상 정보와 자막을 "
                    "수집하고 있습니다..."
                ):
                    summary_video_details = (
                        get_video_details(
                            summary_video_id,
                            api_key,
                        )
                    )

                    if (
                        summary_video_details
                        is None
                    ):
                        raise ValueError(
                            "영상 정보를 찾지 "
                            "못했습니다."
                        )

                    (
                        summary_source_text,
                        summary_source_label,
                    ) = prepare_source_text(
                        video_id=(
                            summary_video_id
                        ),
                        video_details=(
                            summary_video_details
                        ),
                        manual_transcript=(
                            manual_transcript
                        ),
                        uploaded_subtitle=(
                            uploaded_subtitle
                        ),
                    )

                    quick_tfidf_result = (
                        analyze_tfidf(
                            source_text=(
                                summary_source_text
                            ),
                            sentence_count=(
                                summary_sentence_count
                            ),
                        )
                    )

                with st.spinner(
                    "AI가 영상 내용을 "
                    "요약하고 있습니다..."
                ):
                    quick_ai_summary = (
                        generate_ai_summary(
                            video_title=(
                                summary_video_details[
                                    "title"
                                ]
                            ),
                            extractive_summary=(
                                quick_tfidf_result[
                                    "extractive_summary"
                                ]
                            ),
                            keywords=(
                                quick_tfidf_result[
                                    "keywords"
                                ]
                            ),
                            groq_api_key=(
                                groq_api_key
                            ),
                        )
                    )

                st.success(
                    "영상 요약 완료"
                )

                st.markdown(
                    "### "
                    + summary_video_details[
                        "title"
                    ]
                )

                st.caption(
                    "분석 데이터: "
                    + summary_source_label
                )

                if quick_ai_summary["success"]:
                    st.write(
                        quick_ai_summary["text"]
                    )

                    st.caption(
                        "✨ AI가 핵심 문장을 "
                        "바탕으로 자연스럽게 "
                        "재구성한 설명입니다."
                    )

                else:
                    st.write(
                        quick_tfidf_result[
                            "extractive_summary"
                        ]
                    )

                    st.warning(
                        "AI 요약을 사용할 수 "
                        "없어 자막에서 추출한 "
                        "문장을 그대로 "
                        "표시했습니다."
                    )

                    st.write(
                        "오류 유형: "
                        + quick_ai_summary[
                            "error_type"
                        ]
                    )

                    st.code(
                        quick_ai_summary[
                            "error_message"
                        ]
                        or "오류 메시지가 "
                        "없습니다."
                    )

                with st.expander(
                    "핵심 키워드 보기"
                ):
                    st.dataframe(
                        quick_tfidf_result[
                            "keyword_df"
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                with st.expander(
                    "원문 근거 문장 보기 "
                    "(TF-IDF 추출 원본)"
                ):
                    st.write(
                        quick_tfidf_result[
                            "extractive_summary"
                        ]
                    )

            except Exception as error:
                st.error(
                    "영상 요약 실패"
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
                    groq_api_key=(
                        groq_api_key
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
                    groq_api_key=(
                        groq_api_key
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
                "유사도 추천 완료"
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
# 15-1. 분석 결과 탭
# =========================================================

tab_dashboard, tab_tfidf, tab_compare, tab_recommend, tab_export = st.tabs([
    "📌 대시보드",
    "📊 TF-IDF 분석",
    "🔬 유사도 비교",
    "🎯 추천 영상",
    "📥 데이터 내보내기",
])
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

    ai_summary_result = tfidf_result.get(
        "ai_summary_result",
        {"success": False},
    )

    if ai_summary_result.get("success"):
        st.write(
            ai_summary_result["text"]
        )

        st.caption(
            "✨ AI가 핵심 문장을 바탕으로 "
            "자연스럽게 재구성한 설명입니다."
        )

    else:
        st.write(
            tfidf_result["extractive_summary"]
        )

        st.caption(
            "⚠️ AI 요약을 사용할 수 없어 "
            "자막에서 추출한 문장을 그대로 "
            "표시했습니다. (사유: "
            + ai_summary_result.get(
                "error_type", "알 수 없음"
            )
            + ")"
        )

    with st.expander(
        "원문 근거 문장 보기 (TF-IDF 추출 원본)"
    ):
        st.write(
            tfidf_result["extractive_summary"]
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
        st.divider()

        st.header(
            "🎯 유사도 기반 추천"
        )

        st.write(
            "사용한 검색어: `"
            + st.session_state[
                "search_query"
            ]
            + "`"
        )

        method_key = selected_similarity_method
        method_label = SIMILARITY_METHODS[method_key]

        similarity_column_map = (
            recommendation_result[
                "similarity_column_map"
            ]
        )

        (
            absolute_column,
            relative_column,
            relative_percent_column,
        ) = similarity_column_map[method_key]

        maximum_similarity = (
            recommendation_result[
                "maximum_similarity_map"
            ][method_key]
        )

        recommendations = (
            recommendation_result[
                "recommendations_by_method"
            ][method_key]
        )

        st.subheader(
            "선택한 방식: " + method_label
        )

        summary_metric_1, summary_metric_2 = (
            st.columns(2)
        )

        summary_metric_1.metric(
            "최고 원본 유사도",
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
                "유사도가 0보다 큰 "
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
                        method_label + " (원본)",
                        format(
                            row[absolute_column],
                            ".4f",
                        ),
                    )

                    score_column_2.metric(
                        method_label + " (상대)",
                        format(
                            row[
                                relative_percent_column
                            ],
                            ".1f",
                        )
                        + "%",
                    )

                    with st.expander(
                        "다른 유사도 방식 비교 보기"
                    ):
                        (
                            compare_col_1,
                            compare_col_2,
                            compare_col_3,
                            compare_col_4,
                        ) = st.columns(4)

                        compare_col_1.metric(
                            "코사인",
                            format(
                                row[
                                    "cosine_similarity"
                                ],
                                ".4f",
                            ),
                        )

                        compare_col_2.metric(
                            "유클리드",
                            format(
                                row[
                                    "euclidean_similarity"
                                ],
                                ".4f",
                            ),
                        )

                        compare_col_3.metric(
                            "자카드",
                            format(
                                row[
                                    "jaccard_similarity"
                                ],
                                ".4f",
                            ),
                        )

                        compare_col_4.metric(
                            "통합",
                            format(
                                row[
                                    "combined_similarity"
                                ],
                                ".4f",
                            ),
                        )

                    progress_value = float(
                        row[relative_column]
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

                    with st.expander(
                        "💡 이 영상이 추천된 이유"
                    ):
                        candidate = recommendation_result[
                            "candidate_lookup"
                        ][str(row["video_id"])]

                        reason = explain_recommendation(
                            source_details=st.session_state.source_details,
                            tfidf_result=st.session_state.tfidf_result,
                            candidate=candidate,
                            recommendation_row=row,
                            method_key=method_key,
                            recommendation_result=recommendation_result,
                        )

                        st.markdown("**공통 핵심 키워드**")

                        if reason["common_keywords"]:
                            st.write(
                                ", ".join(reason["common_keywords"])
                            )
                        else:
                            st.write(
                                "제목 및 설명에서 공통 토큰을 찾지 못했습니다. "
                                "점수와 원문을 추가로 확인해 주세요."
                            )

                        st.markdown("**유사도 방식별 점수**")

                        score_cols = st.columns(4)

                        for col, method in zip(
                            score_cols,
                            ["cosine", "euclidean", "jaccard", "combined"],
                        ):
                            col.metric(
                                reason["method_labels"][method],
                                f'{reason["scores"][method]:.4f}',
                                help="원본 유사도 점수이며 상대 점수가 아닙니다.",
                            )

                        st.markdown("**다른 방식에서의 순위**")

                        rank_text = " · ".join(
                            (
                                f"{SIMILARITY_METHODS[m]}: {rank2}위"
                                if rank2 is not None
                                else f"{SIMILARITY_METHODS[m]}: 순위 없음"
                            )
                            for m, rank2 in reason["rank_positions"].items()
                        )

                        st.write(rank_text)

                        st.caption(
                            "추천 근거는 현재 수집된 제목·설명·추출 키워드와 "
                            "유사도 점수를 기반으로 계산됩니다. "
                            "공통 키워드가 적더라도 표현이 다르거나 "
                            "설명이 짧으면 유사도가 낮게 나타날 수 있습니다."
                        )

                    st.write(
                        "조회수: "
                        + format(int(row["views"]), ",")
                        + " | 좋아요: "
                        + format(int(row["likes"]), ",")
                    )

            st.subheader(
                method_label + " 비교"
            )

            relative_chart = (
                shown_recommendations[
                    [
                        "title",
                        relative_column,
                    ]
                ]
                .set_index("title")
            )

            st.bar_chart(
                relative_chart
            )

            st.subheader(
                method_label + " (원본값) 비교"
            )

            absolute_chart = (
                shown_recommendations[
                    [
                        "title",
                        absolute_column,
                    ]
                ]
                .set_index("title")
            )

            st.bar_chart(
                absolute_chart
            )

            st.subheader(
                "4가지 유사도 방식 종합 비교 "
                "(상위 추천 기준)"
            )

            multi_method_chart = (
                shown_recommendations[
                    [
                        "title",
                        "cosine_similarity",
                        "euclidean_similarity",
                        "jaccard_similarity",
                        "combined_similarity",
                    ]
                ]
                .rename(
                    columns={
                        "cosine_similarity": (
                            "코사인"
                        ),
                        "euclidean_similarity": (
                            "유클리드"
                        ),
                        "jaccard_similarity": (
                            "자카드"
                        ),
                        "combined_similarity": (
                            "통합"
                        ),
                    }
                )
                .set_index("title")
            )

            st.bar_chart(
                multi_method_chart
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
                        method_key
                        + "_recommendations.csv"
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
