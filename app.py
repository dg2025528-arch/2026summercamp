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
    page_title="YouTube TF-IDF 분석 및 상대 유사도 추천",
    page_icon="🎬",
    layout="wide",
)

MAX_TFIDF_FEATURES = 1500
MAX_TRANSCRIPT_CHARACTERS = 30000

# 절대 기준을 매우 낮게 두되 0에 가까운 결과는 제외합니다.
ABSOLUTE_MINIMUM_COSINE_SIMILARITY = 0.005

# 최고 유사도 후보 대비 35% 이상인 후보를 추천합니다.
RELATIVE_SIMILARITY_THRESHOLD = 0.35

# 최고 유사도 자체가 이 값보다 낮으면 품질 경고를 표시합니다.
LOW_SIMILARITY_WARNING_THRESHOLD = 0.05


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
    if manual_transcript.strip():
        return {
            "text": limit_text(
                manual_transcript
            ),
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

    transcript_result = get_transcript(
        video_id
    )

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
            "파일을 업로드하세요."
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
