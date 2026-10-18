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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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
}

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
MAX_KEYWORDS = 5   # 핵심어 최대 개수
GROQ_MODEL_NAME = "openai/gpt-oss-20b"


# =========================================================
# 2. 한국어 불용어
# =========================================================

KOREAN_STOP_WORDS = {
    "이", "그", "저", "것", "수", "등", "들", "및", "더",
    "를", "을", "은", "는", "가", "에", "의", "와", "과",
    "로", "으로", "에서", "에게", "한", "하다", "합니다",
    "있다", "있는", "있습니다", "없다", "그리고", "하지만",
    "또한", "때문", "대한", "대해", "통해", "오늘",
    "여러분", "제가", "저희", "우리", "이제", "정말", "좀",
    "잘", "거", "게", "건", "입니다", "됩니다", "하면",
    "해서", "하는", "하고", "같은", "이런", "그런", "그래서",
    "하는데", "보면", "한번", "아니", "그러면", "그냥",
    "지금", "여기", "이렇게", "저렇게", "뭐", "또", "아주",
    "많이", "있고", "없고", "위해", "관련", "경우", "이번",
    "근데", "저는", "저도", "일단", "어떤", "항상", "그냥", "진짜",
    "제가", "여러분", "이제", "그래서", "그리고", "그런데", "하지만",
    "이런", "저런", "그런", "그게", "이게", "저게", "너무", "정말",
    "약간", "조금", "많이", "계속", "다시", "사실", "보통", "바로",
    "우리", "이렇게", "그렇게", "어떻게", "하는", "있는", "같은",
    "그러면", "그럼", "이거", "저거", "그거", "아니", "뭔가", "제일",
    "가장", "한번", "오늘", "지금", "부분",
}


# =========================================================
# 3. 텍스트 전처리
# =========================================================

def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*(음악|박수|웃음)[^\)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def limit_text(text):
    return clean_text(text)[:MAX_TRANSCRIPT_CHARACTERS]


def tokenize_mixed_text(text):
    text = clean_text(text).lower()
    tokens = re.findall(r"[가-힣]{2,}|[a-zA-Z][a-zA-Z0-9_-]{1,}", text)

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
    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("자막 파일의 인코딩을 읽지 못했습니다.")


def clean_subtitle_file(text):
    text = str(text)

    text = re.sub(r"^\s*WEBVTT.*?$", " ", text,
                  flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(
        r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}.*$",
        " ", text, flags=re.MULTILINE,
    )
    text = re.sub(
        r"\d{1,2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}[,.]\d{3}.*$",
        " ", text, flags=re.MULTILINE,
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
    decoded_text = decode_uploaded_file(uploaded_file)
    subtitle_text = clean_subtitle_file(decoded_text)
    if len(tokenize_mixed_text(subtitle_text)) < 5:
        raise ValueError("업로드한 자막의 텍스트가 너무 짧습니다.")
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
    hostname = (parsed.hostname or "").lower().replace("www.", "")

    if hostname == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if len(video_id) == 11:
            return video_id

    if hostname in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id and len(video_id) == 11:
                return video_id

        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            video_id = path_parts[1]
            if len(video_id) == 11:
                return video_id

    match = re.search(
        r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})",
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
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def calculate_days_since_upload(published_at):
    if not published_at:
        return 1
    published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max((now - published_date).days, 1)


def get_thumbnail(snippet):
    thumbnails = snippet.get("thumbnails", {})
    for size in ("high", "medium", "default"):
        if size in thumbnails:
            return thumbnails[size].get("url", "")
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def get_video_details(video_id, api_key):
    youtube = get_youtube_client(api_key)

    response = (
        youtube.videos()
        .list(part="snippet,statistics", id=video_id)
        .execute()
    )

    items = response.get("items", [])
    if not items:
        return None

    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})

    published_at = snippet.get("publishedAt", "")
    views = int(statistics.get("viewCount", 0))
    likes = int(statistics.get("likeCount", 0))
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
        "url": "https://www.youtube.com/watch?v=" + video_id,
    }


# =========================================================
# 6. YouTube 자막
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_transcript(video_id):
    """youtube-transcript-api 1.2.3 기준입니다."""

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
        result["error_message"] = str(error) or repr(error)
        return result

    if not transcripts:
        result = empty_result.copy()
        result["error_type"] = "EmptyTranscriptList"
        result["error_message"] = "사용 가능한 자막 목록이 비어 있습니다."
        return result

    def transcript_priority(transcript):
        language_code = str(getattr(transcript, "language_code", "")).lower()
        generated = bool(getattr(transcript, "is_generated", False))

        if language_code.startswith("ko"):
            return 1 if generated else 0
        if language_code.startswith("en"):
            return 3 if generated else 2
        return 5 if generated else 4

    transcripts = sorted(transcripts, key=transcript_priority)

    fetch_errors = []

    for transcript in transcripts:
        try:
            fetched = transcript.fetch()

            text_parts = []
            for snippet in fetched:
                snippet_text = getattr(snippet, "text", "")
                if snippet_text:
                    text_parts.append(str(snippet_text).replace("\n", " "))

            transcript_text = limit_text(" ".join(text_parts))

            if len(tokenize_mixed_text(transcript_text)) >= 5:
                return {
                    "success": True,
                    "text": transcript_text,
                    "language": str(getattr(transcript, "language_code", "unknown")),
                    "generated": bool(getattr(transcript, "is_generated", False)),
                    "translated": False,
                    "error_type": "",
                    "error_message": "",
                }

        except Exception as error:
            fetch_errors.append(type(error).__name__ + ": " + str(error))

    result = empty_result.copy()
    result["error_type"] = "TranscriptFetchFailed"
    result["error_message"] = "\n".join(fetch_errors)
    return result


# =========================================================
# 7. 입력 영상 텍스트 준비
# =========================================================

def prepare_source_text(video_id, video_details, manual_transcript, uploaded_subtitle):
    if manual_transcript.strip():
        text = limit_text(manual_transcript)
        if len(tokenize_mixed_text(text)) < 5:
            raise ValueError("직접 입력한 자막이 너무 짧습니다.")
        return text, "직접 입력한 자막"

    if uploaded_subtitle is not None:
        text = read_uploaded_subtitle(uploaded_subtitle)
        return text, "업로드 자막: " + uploaded_subtitle.name

    transcript_result = get_transcript(video_id)

    if transcript_result["success"]:
        if transcript_result["translated"]:
            label = "YouTube 번역 자막"
        elif transcript_result["generated"]:
            label = "YouTube 자동 생성 자막"
        else:
            label = "YouTube 수동 등록 자막"
        return transcript_result["text"], label

    fallback_text = clean_text(
        video_details["title"] + " " + video_details["title"]
        + " " + video_details["description"]
    )

    if len(tokenize_mixed_text(fallback_text)) < 5:
        raise ValueError(
            "자막 수집에 실패했고 제목·설명도 분석하기에 너무 짧습니다. "
            "자막을 직접 입력하거나 파일을 업로드하세요. "
            "자막 오류: " + transcript_result["error_type"]
        )

    return fallback_text, "YouTube 자막 수집 실패: 제목·설명 사용"


# =========================================================
# 8. TF-IDF 분석
# =========================================================

def analyze_tfidf(source_text, sentence_count, keyword_count=MAX_KEYWORDS):
    sentences = split_sentences(source_text)
    need = int(sentence_count)
    if len(sentences) < need:
        flat = clean_text(source_text)
        size = max(15, len(flat) // need)
        chunks = [flat[i:i + size].strip() for i in range(0, len(flat), size)]
        chunks = [c for c in chunks if len(c) >= 15]
        if len(chunks) > len(sentences):
            sentences = chunks
    if not sentences:
        raise ValueError("분석 가능한 문장을 찾지 못했습니다.")

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
        raise ValueError("TF-IDF 단어 벡터가 비어 있습니다.")

    feature_names = vectorizer.get_feature_names_out()
    sentence_scores = np.asarray(matrix.sum(axis=1)).ravel()

    summary_count = min(int(sentence_count), len(sentences))
    top_sentence_indices = np.argsort(sentence_scores)[::-1][:summary_count]
    ordered_summary_indices = sorted(top_sentence_indices.tolist())

    extractive_summary = " ".join(sentences[i] for i in ordered_summary_indices)

    sentence_rows = []
    for rank, index in enumerate(top_sentence_indices, start=1):
        sentence_rows.append({
            "순위": rank,
            "문장 번호": int(index + 1),
            "TF-IDF 점수": round(float(sentence_scores[index]), 6),
            "문장": sentences[index],
        })
    sentence_score_df = pd.DataFrame(sentence_rows)

    total_term_scores = np.asarray(matrix.sum(axis=0)).ravel()
    keyword_indices = np.argsort(total_term_scores)[::-1][:keyword_count]

    keywords = []
    keyword_rows = []
    for rank, index in enumerate(keyword_indices, start=1):
        keyword = str(feature_names[index])
        keywords.append(keyword)
        keyword_rows.append({
            "순위": rank,
            "단어 또는 구": keyword,
            "TF-IDF 점수": round(float(total_term_scores[index]), 6),
        })
    keyword_df = pd.DataFrame(keyword_rows)

    feature_columns = []
    for index, feature_name in enumerate(feature_names):
        safe_name = re.sub(r"[^가-힣A-Za-z0-9_]+", "_", str(feature_name))
        feature_columns.append("tfidf_" + str(index).zfill(4) + "_" + safe_name)

    sentence_feature_df = pd.DataFrame(matrix.toarray(), columns=feature_columns)
    sentence_feature_df.insert(0, "sentence", sentences)
    sentence_feature_df.insert(1, "sentence_score", sentence_scores)

    return {
        "extractive_summary": extractive_summary,
        "keywords": keywords,
        "keyword_df": keyword_df,
        "sentence_score_df": sentence_score_df,
        "sentence_feature_df": sentence_feature_df,
    }


# =========================================================
# 9. 후보 검색
# =========================================================

def build_search_query(video_title, keywords, custom_query):
    if custom_query.strip():
        return clean_text(custom_query)[:100]

    title_tokens = tokenize_mixed_text(video_title)

    keyword_tokens = []
    for keyword in keywords:
        for part in str(keyword).split():
            part = clean_text(part)
            if len(part) >= 2 and part not in keyword_tokens:
                keyword_tokens.append(part)

    query_tokens = []
    for token in title_tokens[:3] + keyword_tokens[:3]:
        if token not in query_tokens:
            query_tokens.append(token)

    return " ".join(query_tokens)[:100]


@st.cache_data(ttl=3600, show_spinner=False)
def search_candidates(search_query, api_key, maximum_results):
    youtube = get_youtube_client(api_key)

    search_response = (
        youtube.search()
        .list(
            part="snippet",
            q=search_query,
            type="video",
            maxResults=min(int(maximum_results), 50),
            order="relevance",
            safeSearch="moderate",
        )
        .execute()
    )

    video_ids = []
    for item in search_response.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if video_id:
            video_ids.append(video_id)

    if not video_ids:
        return []

    details_response = (
        youtube.videos()
        .list(part="snippet,statistics", id=",".join(video_ids))
        .execute()
    )

    item_by_id = {
        item.get("id", ""): item
        for item in details_response.get("items", [])
    }

    candidates = []
    for video_id in video_ids:
        item = item_by_id.get(video_id)
        if item is None:
            continue

        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})

        title = clean_text(snippet.get("title", ""))
        description = clean_text(snippet.get("description", ""))
        comparison_text = clean_text(
            title + " " + title + " " + title + " " + description
        )

        if len(tokenize_mixed_text(comparison_text)) < 2:
            continue

        published_at = snippet.get("publishedAt", "")
        views = int(statistics.get("viewCount", 0))
        likes = int(statistics.get("likeCount", 0))
        days = calculate_days_since_upload(published_at)

        candidates.append({
            "video_id": video_id,
            "title": title,
            "description": description,
            "comparison_text": comparison_text,
            "channel_title": snippet.get("channelTitle", ""),
            "published_at": published_at,
            "views": views,
            "likes": likes,
            "daily_views": views / days,
            "thumbnail": get_thumbnail(snippet),
            "url": "https://www.youtube.com/watch?v=" + video_id,
        })

    return candidates


# =========================================================
# 10. 유사도 함수 + 권장 방식 판단
# =========================================================

def calculate_jaccard_similarity(source_text, candidate_texts):
    source_tokens = set(tokenize_mixed_text(source_text))

    similarities = []
    for candidate_text in candidate_texts:
        candidate_tokens = set(tokenize_mixed_text(candidate_text))
        union_tokens = source_tokens | candidate_tokens
        if not union_tokens:
            similarities.append(0.0)
        else:
            similarities.append(
                len(source_tokens & candidate_tokens) / len(union_tokens)
            )

    return np.asarray(similarities, dtype=float)


def convert_distance_to_similarity(distances):
    return 1.0 / (1.0 + np.asarray(distances, dtype=float))


def method_columns(key):
    return (f"{key}_similarity", f"{key}_relative", f"{key}_relative_percent")


def build_source_comparison_text(source_details, tfidf_result):
    return clean_text(
        " ".join([source_details["title"]] * 3)
        + " " + tfidf_result["extractive_summary"]
        + " " + " ".join(tfidf_result["keywords"])
    )


def recommend_similarity_method(source_text, candidate_texts):
    source_n = len(tokenize_mixed_text(source_text))
    cand_ns = [len(tokenize_mixed_text(t)) for t in candidate_texts] or [0]
    median_n = float(np.median(cand_ns))
    ratio = median_n / source_n if source_n else 0.0

    if source_n >= 150 and median_n >= 150:
        best = "cosine"
        reason = (
            f"입력({source_n}토큰)과 후보(중앙값 {median_n:.0f}토큰)가 모두 긴 "
            "텍스트라 단어 중요도를 반영하는 코사인이 가장 적합합니다."
        )
    elif ratio < 0.4 or ratio > 2.5:
        best = "cosine"
        reason = (
            f"입력 {source_n}토큰, 후보 중앙값 {median_n:.0f}토큰으로 길이 차이가 "
            "큽니다. 자카드는 합집합이 커져 점수가 눌리므로 코사인을 권장합니다."
        )
    elif median_n < 30 and source_n < 60:
        best = "jaccard"
        reason = (
            f"입력 {source_n}토큰, 후보 중앙값 {median_n:.0f}토큰으로 모두 짧습니다. "
            "짧은 글은 TF-IDF 가중치가 불안정하므로 자카드를 우선하세요."
        )
    else:
        best = "cosine"
        reason = (
            f"입력 {source_n}토큰, 후보 중앙값 {median_n:.0f}토큰으로 길이가 비슷한 "
            "중간 길이 텍스트라 코사인을 기본으로 권장합니다."
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

    source_vector = matrix[0:1]
    candidate_vectors = matrix[1:]

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
        abs_col, rel_col, pct_col = method_columns(key)
        maximum = float(np.max(values))
        relative = values / maximum if maximum > 0 else np.zeros(len(values))

        all_results_df[abs_col] = values.astype(float)
        all_results_df[rel_col] = relative.astype(float)
        all_results_df[pct_col] = (relative * 100.0).astype(float)

        similarity_column_map[key] = (abs_col, rel_col, pct_col)
        maximum_similarity_map[key] = maximum

    def filter_for_method(key):
        abs_col, _, _ = similarity_column_map[key]

        filtered = all_results_df[
            all_results_df[abs_col] >= ABSOLUTE_MINIMUM_SIMILARITY
        ].copy()

        if filtered.empty and maximum_similarity_map[key] > 0:
            filtered = all_results_df.loc[[all_results_df[abs_col].idxmax()]].copy()

        return filtered.sort_values(abs_col, ascending=False).reset_index(drop=True)

    recommendations_by_method = {
        key: filter_for_method(key) for key in SIMILARITY_METHODS
    }

    feature_columns = [
        "tfidf_" + str(i).zfill(4) + "_"
        + re.sub(r"[^가-힣A-Za-z0-9_]+", "_", str(name))
        for i, name in enumerate(vectorizer.get_feature_names_out())
    ]
    candidate_feature_df = pd.DataFrame(
        candidate_vectors.toarray(), columns=feature_columns
    )

    metadata_columns = [
        "video_id", "title", "channel_title", "url",
        "views", "likes", "daily_views",
    ]
    for key in SIMILARITY_METHODS:
        metadata_columns += list(similarity_column_map[key])

    orange_df = pd.concat(
        [
            all_results_df[metadata_columns].reset_index(drop=True),
            candidate_feature_df.reset_index(drop=True),
        ],
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


# =========================================================
# 11. 추천 이유 + 보조 함수
# =========================================================

def explain_recommendation(
    source_details,
    tfidf_result,
    candidate,
    recommendation_row,
    method_key,
    recommendation_result,
):
    source_tokens = set(
        tokenize_mixed_text(recommendation_result["source_comparison_text"])
    )
    candidate_tokens = set(tokenize_mixed_text(candidate["comparison_text"]))
    common = source_tokens & candidate_tokens

    prioritized = [w for w in tfidf_result["keywords"] if w in common]
    if not prioritized:
        prioritized = sorted(common, key=len, reverse=True)
    prioritized = prioritized[:MAX_KEYWORDS]

    scores = {
        key: recommendation_row[f"{key}_similarity"]
        for key in SIMILARITY_METHODS
    }

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
            f"입력 영상의 핵심어가 이 영상의 제목·설명에도 나옵니다: "
            + ", ".join(prioritized)
        )
    else:
        reasons.append("공통 핵심어는 적지만 전체 단어 분포가 일부 비슷합니다.")

    reasons.append(
        f"{SIMILARITY_METHODS[method_key]} 점수 {scores[method_key]:.4f}, "
        f"후보 {len(all_results)}개 중 {rank_positions[method_key]}위입니다."
    )

    ranks = [r for r in rank_positions.values() if r is not None]
    if ranks and max(ranks) <= 3:
        reasons.append("세 가지 유사도 모두 상위 3위 안이라 일관된 추천입니다.")
    elif ranks and max(ranks) - min(ranks) >= 5:
        reasons.append("방식별 순위 차이가 커서 한 기준에만 의존한 추천일 수 있습니다.")

    if method_key != advice["best"]:
        reasons.append(
            f"이번 데이터에는 {SIMILARITY_METHODS[advice['best']]}가 권장 방식입니다."
        )

    if candidate.get("daily_views", 0) > source_details.get("daily_views", 0):
        reasons.append(
            f"하루 평균 조회수 {candidate['daily_views']:,.0f}회로 "
            "입력 영상보다 활발히 시청되고 있습니다."
        )

    if candidate.get("channel_title") == source_details.get("channel_title"):
        reasons.append("입력 영상과 같은 채널입니다.")

    return {
        "common_keywords": prioritized,
        "scores": scores,
        "rank_positions": rank_positions,
        "reasons": reasons,
    }


def create_pair_tfidf_table(source_details, tfidf_result, candidate):
    source_text = build_source_comparison_text(source_details, tfidf_result)

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

    matrix = vectorizer.fit_transform([source_text, candidate["comparison_text"]])
    source_values = matrix[0].toarray().ravel()
    candidate_values = matrix[1].toarray().ravel()

    detail_df = pd.DataFrame({
        "단어": vectorizer.get_feature_names_out(),
        "IDF": vectorizer.idf_,
        "입력 영상 TF-IDF": source_values,
        "후보 영상 TF-IDF": candidate_values,
        "TF-IDF 합계": source_values + candidate_values,
    })

    return detail_df.sort_values("TF-IDF 합계", ascending=False).head(30)


def to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8-sig")


# =========================================================
# 12. AI 요약  ★ 원래 쓰던 generate_ai_summary가 있으면
#     이 함수만 그걸로 교체하세요. (반환 형식 유지 필수)
# =========================================================

def generate_ai_summary(video_title, extractive_summary, keywords,
                        groq_api_key, source_text="", sentence_count=5):
    result = {"success": False, "text": "", "error_type": "", "error_message": ""}

    if not groq_api_key:
        result["error_type"] = "NoGroqKey"
        result["error_message"] = "GROQ_API_KEY가 설정되지 않았습니다."
        return result

    n = int(sentence_count)
    body = source_text[:12000] if source_text else extractive_summary

    system_prompt = (
        "당신은 영상 내용을 사실 그대로 정리하는 한국어 요약가입니다. "
        "제공된 자막에 실제로 나온 내용만 정리하세요. "
        "감상평, 평가, 의견은 쓰지 마세요. 없는 내용을 지어내지 마세요. "
        "자막이 외국어여도 자연스러운 한국어로 쓰세요."
    )
    user_prompt = (
        f"[영상 제목]\n{video_title}\n\n"
        f"[자막/본문]\n{body}\n\n"
        f"[핵심어 참고]\n{', '.join(keywords)}\n\n"
        f"위 내용을 정확히 {n}개의 문장으로 요약하세요.\n"
        "- 한 문단은 영상에서 다룬 순서대로, 한 줄에 한 문장씩 줄바꿈으로 구분합니다.\n"
        "- 번호, 불릿, 제목, 한 줄 요약, 결론, 서두 문구를 넣지 않습니다.\n"
        "- 인사말, 구독 요청, 광고는 제외합니다.\n"
        "- 자막을 그대로 복사하지 말고 자신의 말로 풀어 씁니다."
    )

    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            temperature=0.2,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        result["text"] = response.choices[0].message.content.strip()
        result["success"] = True
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)

    return result

# =========================================================
# 13. 세션 상태 + 분석 실행
# =========================================================

SESSION_DEFAULTS = {
    "source_details": None,
    "source_text": "",
    "source_label": "",
    "tfidf_result": None,
    "recommendation_result": None,
    "search_query": "",
}

for session_key, default_value in SESSION_DEFAULTS.items():
    if session_key not in st.session_state:
        st.session_state[session_key] = default_value


def reset_session_results():
    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value


def perform_tfidf_analysis(
    youtube_url,
    api_key,
    manual_transcript,
    uploaded_subtitle,
    sentence_count,
    groq_api_key,
):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("올바른 YouTube URL을 입력하세요.")

    video_details = get_video_details(video_id, api_key)
    if video_details is None:
        raise ValueError("영상 정보를 찾지 못했습니다.")

    source_text, source_label = prepare_source_text(
        video_id=video_id,
        video_details=video_details,
        manual_transcript=manual_transcript,
        uploaded_subtitle=uploaded_subtitle,
    )

    tfidf_result = analyze_tfidf(
        source_text=source_text,
        sentence_count=sentence_count,
    )

    tfidf_result["ai_summary_result"] = generate_ai_summary(
        video_title=video_details["title"],
        extractive_summary=tfidf_result["extractive_summary"],
        keywords=tfidf_result["keywords"],
        groq_api_key=groq_api_key,
        source_text=source_text,
        sentence_count=sentence_count,
    )

    st.session_state.source_details = video_details
    st.session_state.source_text = source_text
    st.session_state.source_label = source_label
    st.session_state.tfidf_result = tfidf_result

    return video_id, video_details, tfidf_result

# =========================================================
# 14. 스타일 (글래스모피즘 + 애니메이션)
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    html {scroll-behavior: smooth;}

    @keyframes aurora {
        0%   {background-position: 0% 50%;}
        50%  {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    @keyframes fadeUp {
        from {opacity: 0; transform: translateY(24px);}
        to   {opacity: 1; transform: translateY(0);}
    }

    /* 폰트: 각진 고딕 + 글자색 전부 검정 */
   html, body, .stApp {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
.stApp *:not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="material-symbols"]) {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif !important;
}
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp div {
    color: #111111;
}
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded" !important;
}

    /* 배경: 오로라 그라데이션 */
    .stApp {
        background: linear-gradient(120deg,
            #c9b6ff, #a8c8ff, #ffc2e2, #ffd9b8, #b8f0e0, #c9b6ff);
        background-size: 400% 400%;
        animation: aurora 28s ease infinite;
    }
    [data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 1200px; padding-top: 2rem;}

    /* 카드: 거의 불투명한 흰색이라 글씨가 또렷함 */
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(255,255,255,.75);
        border-right: 1px solid rgba(255,255,255,.9);
    }
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stExpander"],
    [data-testid="stMetric"],
    [data-testid="stForm"] {
        background: rgba(255,255,255,.82);
        border: 1px solid rgba(255,255,255,.95) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 22px rgba(80,60,160,.12);
    }
    [data-testid="stMetric"] {padding: .6rem 1rem;}

    [data-testid="stVerticalBlockBorderWrapper"] {
        animation: fadeUp .6s ease both;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(80,60,160,.22);
    }

    .section-title {
        font-size: 1.7rem; font-weight: 700; margin: 2.2rem 0 .8rem;
        padding: .5rem 1.1rem; border-radius: 12px;
        background: rgba(255,255,255,.85);
        border-left: 6px solid #ff4b4b;
        animation: fadeUp .7s ease both;
    }

    .landing {text-align:center; padding: 3.5rem 1rem 1.5rem;
              animation: fadeUp .8s ease both;}
    .landing h1 {font-size: 2.8rem; font-weight: 700; margin-bottom: .6rem;}
    .landing p {font-size: 1.1rem;}

    /* 버튼: 작고 단정하게 (사이드바 포함) */
    .stButton button, [data-testid="stFormSubmitButton"] button {
        background: rgba(255,255,255,.9);
        border: 1px solid #999; border-radius: 10px;
        font-weight: 500;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(80,60,160,.25);
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%; height: 2.8rem; min-height: 2.8rem;
    }
    [data-testid="stSidebar"] .stButton button p {
        font-size: 1.05rem !important; font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
        background: #ffd0d0; border: 2px solid #ff4b4b;
    }

    .nav-link {
        display:block; text-align:center; text-decoration:none !important;
        font-size: 1.05rem; font-weight: 500; padding: .65rem 0;
        margin-bottom: .5rem; border-radius: 10px;
        background: rgba(255,255,255,.9); border: 1px solid #999;
        transition: transform .15s ease, background .15s ease;
    }
    .nav-link:hover {transform: translateX(5px); background: #ffd0d0;}

    .chip {display:inline-block; padding:3px 12px; margin:3px;
           border-radius:999px; background:#fff; border:1px solid #999;
           font-size:.9rem; font-weight:500;}
    .badge {display:inline-block; padding:3px 12px; border-radius:8px;
            background:#d8f5da; font-weight:700; font-size:.9rem;}

    @media (prefers-reduced-motion: reduce) {
        * {animation: none !important; transition: none !important;}
    }
        /* 표/차트: 흰 배경 박스로 감싸 글씨 가독성 확보 */
    [data-testid="stDataFrame"],
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stVegaLiteChart"] {
        background: #ffffff !important;
        border: 1px solid #cfcfcf;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 4px 14px rgba(80,60,160,.12);
    }
    [data-testid="stDataFrame"] iframe {
        background: #ffffff !important;
        color-scheme: light;
    }

    /* 요약 박스 */
    .summary-box {
        background: rgba(255,255,255,.92);
        border-left: 6px solid #8e2de2;
        border-radius: 12px; padding: 1rem 1.3rem; margin: .6rem 0;
        box-shadow: 0 4px 14px rgba(80,60,160,.12);
    }
    .summary-box h4 {margin: 0 0 .5rem; font-size: 1.15rem;}
    .summary-box ul {margin: 0; padding-left: 1.2rem;}
    .summary-box li {margin-bottom: .45rem; line-height: 1.6;}
    .quote-box {
        background: #fffbe8; border-left: 5px solid #f5b400;
        border-radius: 10px; padding: .8rem 1.1rem; margin: .5rem 0;
        line-height: 1.7;
    }
    /* 사이드바 정리 */
    [data-testid="stSidebar"] {min-width: 250px; max-width: 250px;}
    .side-title {font-size: 1.2rem; font-weight: 700; margin: .3rem 0 .6rem;}
    
    /* 다운로드 버튼: 흰 배경 + 검정 글씨 */
    [data-testid="stDownloadButton"] button {
        background: #ffffff !important;
        border: 1px solid #999 !important;
        border-radius: 10px;
    }
    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span {
        color: #111111 !important;
        font-weight: 500;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #ffd0d0 !important;
    }

    /* 핵심어: 단어마다 따로 떨어진 칩 */
    .chip-row {display: flex; flex-wrap: wrap; gap: 10px; margin: .4rem 0 1rem;}
    .chip-row .chip {margin: 0; padding: 6px 16px; font-size: 1rem;
                     background: #ffffff; border: 1px solid #888;}
    .chip-row {display: flex; flex-wrap: wrap; gap: 14px; margin: .6rem 0 1.2rem;}
    .chip-row .chip {
        margin: 0; padding: 10px 24px; font-size: 1.15rem; font-weight: 500;
        background: #ffffff; border: 1px solid #333; border-radius: 999px;
        box-shadow: none; transition: transform .15s ease, background .15s ease;
    }
    .chip-row .chip:hover {transform: translateY(-3px); background: #ffd0d0;}
        .query-box {
        background: linear-gradient(135deg, #ff4b4b, #8e2de2);
        border-radius: 16px; padding: 1rem 1.4rem; margin: .8rem 0;
        box-shadow: 0 8px 24px rgba(142,45,226,.35);
    }
    .query-box .q-label {font-size: .9rem; font-weight: 700; margin-bottom: .6rem;}
    .query-box .q-row {display: flex; flex-wrap: wrap; gap: 10px;}
    .query-box .q-chip {
        padding: 6px 18px; border-radius: 999px; font-size: 1.1rem;
        font-weight: 700; background: #ffffff;
        border: 1px solid #ffffff;
    }
    .query-box .q-label {color: #ffffff !important;}
    .query-box .q-chip {color: #111111 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)
# =========================================================
# 15. 상태 + 설정(사이드바)
# =========================================================

if "entered" not in st.session_state:
    st.session_state.entered = False
if "settings" not in st.session_state:
    st.session_state.settings = {
        "sentence_count": 5,
        "candidate_count": 20,
        "recommendation_count": 5,
    }
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "input_url" not in st.session_state:
    st.session_state.input_url = ""
if "input_query" not in st.session_state:
    st.session_state.input_query = ""

api_key = get_api_key()
groq_api_key = get_groq_api_key()


def show_error(title, error):
    st.error(title)
    st.write("오류 유형: " + type(error).__name__)
    st.write("오류 내용: " + str(error))
    with st.expander("상세 오류"):
        st.code(traceback.format_exc())


# =========================================================
# 16. 시작 화면
# =========================================================

if not st.session_state.entered:
    st.markdown(
        """
        <div class="landing">
          <h1>🎬 YouTube 요약 · 유사 영상 추천</h1>
          <p>영상 링크를 입력하면 요약, TF-IDF 분석, 비슷한 영상 추천까지 한 번에 보여드립니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 2, 1])
    with center:
        if not api_key:
            st.warning("Streamlit Secrets에 YOUTUBE_API_KEY를 설정하세요.")
        if not groq_api_key:
            st.info("GROQ_API_KEY가 없어 AI 요약 대신 TF-IDF 추출 요약을 사용합니다.")

        with st.form("landing_form"):
            url_in = st.text_input(
                "YouTube 영상 링크",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            query_in = st.text_input(
                "후보 검색어 (선택, 비우면 자동 생성)",
                placeholder="예: 인공지능 교육 활용",
            )
            with st.expander("자막 직접 입력 또는 업로드 (선택)"):
                manual_in = st.text_area("자막 직접 입력", height=120)
                upload_in = st.file_uploader(
                    "TXT, SRT, VTT 자막 파일", type=["txt", "srt", "vtt"]
                )
            go = st.form_submit_button(
                "🚀 입장하기", type="primary", use_container_width=True
            )

        if go:
            if not extract_video_id(url_in):
                st.error("올바른 YouTube 링크를 입력하세요.")
            elif not api_key:
                st.error("YOUTUBE_API_KEY가 필요합니다.")
            else:
                st.session_state.input_url = url_in.strip()
                st.session_state.input_query = query_in.strip()
                st.session_state.input_manual = manual_in
                st.session_state.input_upload = upload_in
                st.session_state.entered = True
                st.session_state.analysis_sig = None
                st.session_state.recommend_sig = None
                st.rerun()

    st.stop()


# =========================================================
# 17. 사이드바 (입장 후): 설정 + 이동 메뉴
# =========================================================

with st.sidebar:
    st.markdown("<div class='side-title'>🎬 메뉴</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <a class="nav-link" href="#sec-info">🎥 영상 정보</a>
        <a class="nav-link" href="#sec-summary">📝 요약</a>
        <a class="nav-link" href="#sec-tfidf">📊 TF-IDF</a>
        <a class="nav-link" href="#sec-similar">🎯 유사도</a>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    with st.expander("⚙️ 설정", expanded=False):
        current = st.session_state.settings
        with st.form("settings_form"):
            new_sentence = st.slider("요약 문장 수", 3, 10, current["sentence_count"])
            new_candidate = st.slider(
                "후보 영상 수", 5, 50, current["candidate_count"], step=5
            )
            new_recommend = st.slider(
                "표시할 추천 수", 1, 10, current["recommendation_count"]
            )
            if st.form_submit_button("적용", use_container_width=True):
                st.session_state.settings = {
                    "sentence_count": new_sentence,
                    "candidate_count": new_candidate,
                    "recommendation_count": new_recommend,
                }
                st.rerun()

    st.divider()
    if st.button("🏠 처음으로", use_container_width=True):
        st.session_state.entered = False
        st.rerun()

settings = st.session_state.settings
summary_sentence_count = settings["sentence_count"]
candidate_count = settings["candidate_count"]
recommendation_count = settings["recommendation_count"]

youtube_url = st.session_state.input_url
custom_search_query = st.session_state.input_query
manual_transcript = st.session_state.get("input_manual", "")
uploaded_subtitle = st.session_state.get("input_upload", None)


# =========================================================
# 18. 분석 자동 실행 (요약 + TF-IDF + 유사도 한 번에)
# =========================================================

def run_analysis():
    return perform_tfidf_analysis(
        youtube_url=youtube_url,
        api_key=api_key,
        manual_transcript=manual_transcript,
        uploaded_subtitle=uploaded_subtitle,
        sentence_count=summary_sentence_count,
        groq_api_key=groq_api_key,
    )


analysis_sig = (
    youtube_url,
    manual_transcript.strip(),
    uploaded_subtitle.name if uploaded_subtitle is not None else "",
    summary_sentence_count,
)
recommend_sig = analysis_sig + (custom_search_query, candidate_count)

try:
    if st.session_state.get("analysis_sig") != analysis_sig:
        with st.spinner("영상 분석 및 AI 요약 중..."):
            run_analysis()
        st.session_state.analysis_sig = analysis_sig
        st.session_state.recommendation_result = None
        st.session_state.recommend_sig = None

    if st.session_state.get("recommend_sig") != recommend_sig:
        source_video_id = extract_video_id(youtube_url)
        details_now = st.session_state.source_details
        tfidf_now = st.session_state.tfidf_result

        search_query = build_search_query(
            details_now["title"], tfidf_now["keywords"], custom_search_query
        )
        if not search_query:
            raise ValueError("후보 검색어를 만들지 못했습니다.")

        with st.spinner("후보 검색 및 유사도 계산 중..."):
            candidates = [
                c
                for c in search_candidates(search_query, api_key, candidate_count)
                if c["video_id"] != source_video_id
            ]
            st.session_state.recommendation_result = (
                calculate_relative_recommendations(
                    details_now, tfidf_now, candidates
                )
            )
        st.session_state.search_query = search_query
        st.session_state.recommend_sig = recommend_sig

except Exception as error:
    st.session_state.analysis_sig = None
    st.session_state.recommend_sig = None
    show_error("분석 실패", error)
    st.stop()


# =========================================================
# 19. 화면: 영상 정보 → 요약 → TF-IDF → 유사도
# =========================================================

source_details = st.session_state.source_details
tfidf_result = st.session_state.tfidf_result
recommendation_result = st.session_state.recommendation_result


def render_method_tab(key):
    label = SIMILARITY_METHODS[key]
    guide = METHOD_GUIDE[key]
    advice = recommendation_result["method_advice"]

    if advice["best"] == key:
        st.markdown(
            "<span class='badge'>✅ 이번 데이터에 권장</span>",
            unsafe_allow_html=True,
        )
        st.caption(advice["reason"])
    else:
        st.caption(
            f"이번 데이터에는 {SIMILARITY_METHODS[advice['best']]}가 권장됩니다."
        )

    with st.expander(f"📐 {label}는 어떤 데이터에 좋은가"):
        st.markdown("- **적합한 데이터**: " + guide["best_for"])
        st.markdown("- **이유**: " + guide["why"])
        st.markdown("- **주의점**: " + guide["weak"])

    abs_col, _, _ = recommendation_result["similarity_column_map"][key]
    maximum = recommendation_result["maximum_similarity_map"][key]
    recommendations = recommendation_result["recommendations_by_method"][key]

    if maximum < LOW_QUALITY_WARNING:
        st.warning("후보 전체의 유사도가 낮아 참고용으로만 보세요.")

    if recommendations.empty:
        st.warning("유사도가 0보다 큰 후보를 찾지 못했습니다.")
        return

    shown = recommendations.head(recommendation_count)

    for rank, row_tuple in enumerate(shown.itertuples(index=False), start=1):
        row = row_tuple._asdict()

        with st.container(border=True):
            img_col, info_col, score_col = st.columns([1.3, 3.5, 1])

            with img_col:
                if row["thumbnail"]:
                    st.image(row["thumbnail"], use_container_width=True)

            with info_col:
                st.markdown(f"**{rank}. [{row['title']}]({row['url']})**")
                st.caption(
                    f"{row['channel_title']} · 조회수 {int(row['views']):,}"
                )

            with score_col:
                st.metric(label.replace(" 유사도", ""), f"{row[abs_col]:.3f}")

            candidate = recommendation_result["candidate_lookup"][str(row["video_id"])]
            reason = explain_recommendation(
                source_details, tfidf_result, candidate,
                row, key, recommendation_result,
            )

            with st.expander("💡 이 영상이 추천된 이유", expanded=False):
                for line in reason["reasons"]:
                    st.markdown("- " + line)

                if reason["common_keywords"]:
                    st.markdown(
                        "".join(
                            f"<span class='chip'>{w}</span>"
                            for w in reason["common_keywords"]
                        ),
                        unsafe_allow_html=True,
                    )

    st.download_button(
        "추천 결과 CSV",
        to_csv_bytes(recommendations),
        f"{key}_recommendations.csv",
        "text/csv",
        key=f"dl_rec_{key}",
    )


# ---------- 🎥 영상 정보 ----------
st.markdown("<div id='sec-info'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🎥 영상 정보</div>", unsafe_allow_html=True)

left, right = st.columns([1, 2])
with left:
    st.video(source_details["url"])
with right:
    st.markdown("### " + source_details["title"])
    st.caption(
        source_details["channel_title"]
        + " · 분석 데이터: "
        + st.session_state.source_label
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("조회수", f"{source_details['views']:,}")
    m2.metric("좋아요", f"{source_details['likes']:,}")
    m3.metric("일평균 조회", f"{source_details['daily_views']:,.0f}")
    query_chips = "".join(
        f"<span class='q-chip'>{w}</span>"
        for w in st.session_state.search_query.split()
    )
    st.markdown(
        "<div class='query-box'><div class='q-label'>🔍 후보 검색어</div>"
        f"<div class='q-row'>{query_chips}</div></div>",
        unsafe_allow_html=True,
    )


# ---------- 📝 요약 ----------
ai = tfidf_result.get("ai_summary_result", {"success": False})

if ai.get("success"):
    items = "".join(
        f"<li>{line.strip()}</li>"
        for line in ai["text"].splitlines()
        if line.strip()
    )
    st.markdown(
        f"<div class='summary-box'><h4>📌 핵심 내용</h4><ul>{items}</ul></div>",
        unsafe_allow_html=True,
    )
else:
    st.warning("AI 요약을 사용할 수 없어 아래 핵심 문장만 표시합니다.")
    st.write("오류 유형: " + ai.get("error_type", "알 수 없음"))
    st.code(ai.get("error_message") or "오류 메시지가 없습니다.")

sentence_df = tfidf_result["sentence_score_df"]
quotes = "".join(
    f"<div class='quote-box'>“{s}”</div>"
    for s in sentence_df.sort_values("문장 번호")["문장"].tolist()
)
st.markdown(
    f"#### 💬 핵심 문장 ({len(sentence_df)}개)"
)
st.markdown(quotes, unsafe_allow_html=True)

st.markdown("**핵심어 (최대 5개)**")
st.markdown(
    "<div class='chip-row'>"
    + "".join(
        f"<span class='chip'>{k}</span>"
        for k in tfidf_result["keywords"][:MAX_KEYWORDS]
    )
    + "</div>",
    unsafe_allow_html=True,
)
# ---------- 📊 TF-IDF ----------
st.markdown("<div id='sec-tfidf'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📊 TF-IDF 분석</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("핵심어")
    st.dataframe(
        tfidf_result["keyword_df"], use_container_width=True, hide_index=True
    )
    st.bar_chart(
        tfidf_result["keyword_df"].set_index("단어 또는 구")["TF-IDF 점수"]
    )
with c2:
    st.subheader("중요 문장")
    st.dataframe(
        tfidf_result["sentence_score_df"],
        use_container_width=True,
        hide_index=True,
    )

e1, e2 = st.columns(2)
e1.download_button(
    "키워드 CSV",
    to_csv_bytes(tfidf_result["keyword_df"]),
    "tfidf_keywords.csv",
    "text/csv",
    use_container_width=True,
)
e2.download_button(
    "문장 벡터 CSV",
    to_csv_bytes(tfidf_result["sentence_feature_df"]),
    "tfidf_sentence_vectors.csv",
    "text/csv",
    use_container_width=True,
)


# ---------- 🎯 유사도 추천 ----------
st.markdown("<div id='sec-similar'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🎯 유사도 추천</div>", unsafe_allow_html=True)

tab_cos, tab_euc, tab_jac = st.tabs(["코사인", "유클리드", "자카드"])
with tab_cos:
    render_method_tab("cosine")
with tab_euc:
    render_method_tab("euclidean")
with tab_jac:
    render_method_tab("jaccard")
