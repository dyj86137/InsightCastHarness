"""InsightCast V1 领域枚举。"""

from __future__ import annotations

from enum import Enum


class Industry(str, Enum):
    """需要追踪的商业行业。"""

    AI = "ai"
    ECOMMERCE = "ecommerce"
    RETAIL = "retail"
    CONSUMER = "consumer"
    FINANCE = "finance"
    CLOUD = "cloud"
    ENTERPRISE_SOFTWARE = "enterprise_software"
    MEDIA = "media"
    HEALTHCARE = "healthcare"
    AUTO = "auto"
    ENERGY = "energy"
    OTHER = "other"


class SourceType(str, Enum):
    """支持的内容来源类型。"""

    YOUTUBE = "youtube"
    BILIBILI = "bilibili"
    VIMEO = "vimeo"
    DAILYMOTION = "dailymotion"
    TWITCH = "twitch"
    PEERTUBE = "peertube"
    PODCAST_RSS = "podcast_rss"
    RSS = "rss"
    WEBSITE = "website"
    MANUAL = "manual"
    OTHER = "other"


class ContentFormat(str, Enum):
    """候选内容的格式。"""

    VIDEO = "video"
    AUDIO = "audio"
    ARTICLE = "article"
    TRANSCRIPT = "transcript"
    UNKNOWN = "unknown"


class InterviewType(str, Enum):
    """规范化后的访谈类内容类型。"""

    INTERVIEW = "interview"
    PODCAST_INTERVIEW = "podcast_interview"
    FIRESIDE_CHAT = "fireside_chat"
    KEYNOTE = "keynote"
    PANEL = "panel"
    EARNINGS_CALL = "earnings_call"
    SPEECH = "speech"
    NEWS_CLIP = "news_clip"
    COMMENTARY = "commentary"
    UNKNOWN = "unknown"


class CandidateStatus(str, Enum):
    """候选项的审核与处理状态。"""

    DISCOVERED = "discovered"
    CLASSIFIED = "classified"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    ARCHIVED = "archived"
    FAILED = "failed"


class InterviewStatus(str, Enum):
    """规范化访谈的处理状态。"""

    NEW = "new"
    TRANSCRIPT_PENDING = "transcript_pending"
    TRANSCRIPT_READY = "transcript_ready"
    SUMMARY_READY = "summary_ready"
    PUSH_READY = "push_ready"
    PUSHED = "pushed"
    ARCHIVED = "archived"
    FAILED = "failed"


class TranscriptStatus(str, Enum):
    """转录文本的可用状态。"""

    UNAVAILABLE = "unavailable"
    PENDING = "pending"
    PARTIAL = "partial"
    READY = "ready"
    FAILED = "failed"


class TranscriptSource(str, Enum):
    """转录文本的来源渠道。"""

    OFFICIAL_CAPTION = "official_caption"
    PLATFORM_CAPTION = "platform_caption"
    SOURCE_METADATA = "source_metadata"
    PODCAST_TRANSCRIPT = "podcast_transcript"
    MEDIA_ARTICLE = "media_article"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class BriefStatus(str, Enum):
    """每日简报的生命周期状态。"""

    DRAFT = "draft"
    READY = "ready"
    SENT = "sent"
    FAILED = "failed"


class BriefSection(str, Enum):
    """每日简报中的分区。"""

    MUST_READ = "must_read"
    WORTH_WATCHING = "worth_watching"
    ARCHIVED = "archived"


class FeedbackType(str, Enum):
    """用户反馈动作。"""

    SAVE = "save"
    IGNORE = "ignore"
    NOT_RELEVANT = "not_relevant"
    NOT_INTERVIEW = "not_interview"
    DUPLICATE = "duplicate"
    SOURCE_UNTRUSTED = "source_untrusted"
    PERSON_MISMATCH = "person_mismatch"
    SUMMARY_POOR = "summary_poor"
    FOLLOW_UP = "follow_up"


class PushChannel(str, Enum):
    """V1 支持的推送渠道。"""

    MARKDOWN = "markdown"
    EMAIL = "email"
    FEISHU = "feishu"
    TELEGRAM = "telegram"


__all__ = [
    "BriefSection",
    "BriefStatus",
    "CandidateStatus",
    "ContentFormat",
    "FeedbackType",
    "Industry",
    "InterviewStatus",
    "InterviewType",
    "PushChannel",
    "SourceType",
    "TranscriptSource",
    "TranscriptStatus",
]
