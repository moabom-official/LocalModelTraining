"""Text cleaning, PII scrub, language detection, near-dup keying."""
from __future__ import annotations

import hashlib
import re
import unicodedata

EMAIL_RE = re.compile(r"[\w\.\-+]+@[\w\.\-]+\.\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w._\-]+")
# Phone: must contain at least one separator so we don't eat plain numbers like "120Hz"
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s])?(?:\d{2,4}[-.\s]){1,3}\d{2,4}")
WS_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")

HANGUL_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
CJK_OTHER_RE = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")
NORMALIZE_KEY_RE = re.compile(r"[^\w가-힣]+")


def detect_lang(text: str) -> str:
    """Cheap heuristic. Returns one of: ko, en, ja_zh, ru, other."""
    if not text:
        return "other"
    ko = len(HANGUL_RE.findall(text))
    en = len(LATIN_RE.findall(text))
    cyr = len(CYRILLIC_RE.findall(text))
    cjk = len(CJK_OTHER_RE.findall(text))
    total = ko + en + cyr + cjk
    if total == 0:
        return "other"
    if ko / total >= 0.2:
        return "ko"
    if cyr / total >= 0.4:
        return "ru"
    if cjk / total >= 0.4:
        return "ja_zh"
    if en / total >= 0.4:
        return "en"
    return "other"


def scrub_pii(text: str) -> str:
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = MENTION_RE.sub("<USER>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    return text


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = scrub_pii(text)
    text = REPEAT_CHAR_RE.sub(r"\1\1\1", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def near_dup_key(text: str) -> str:
    """Stable hash for fuzzy dedup (case + non-alnum collapsed)."""
    norm = NORMALIZE_KEY_RE.sub("", text.lower())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()
