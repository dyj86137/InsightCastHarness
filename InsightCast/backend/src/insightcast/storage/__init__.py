"""InsightCast V1 持久化层。"""

from insightcast.storage.base import (
    DuplicateRecordError,
    InvalidRecordError,
    RecordNotFoundError,
    Repository,
    StorageError,
)
from insightcast.storage.json_store import JsonCollectionStore, default_storage_dir
from insightcast.storage.sqlite_store import (
    SQLiteCollectionStore,
    default_sqlite_path,
    resolve_sqlite_path,
)
from insightcast.storage.repositories import (
    CandidateRepository,
    DailyBriefRepository,
    IndustryProfileRepository,
    InsightCastRepositories,
    InterviewDecisionRepository,
    InterviewRepository,
    InterviewSummaryRepository,
    JsonRepository,
    PersonRepository,
    RunRecordRepository,
    SearchQueryRepository,
    SourceRepository,
    TranscriptRepository,
    UserFeedbackRepository,
    UserInterestRepository,
)


__all__ = [
    "CandidateRepository",
    "DailyBriefRepository",
    "DuplicateRecordError",
    "IndustryProfileRepository",
    "InsightCastRepositories",
    "InterviewDecisionRepository",
    "InterviewRepository",
    "InterviewSummaryRepository",
    "InvalidRecordError",
    "JsonCollectionStore",
    "JsonRepository",
    "PersonRepository",
    "RecordNotFoundError",
    "Repository",
    "RunRecordRepository",
    "SearchQueryRepository",
    "SourceRepository",
    "StorageError",
    "SQLiteCollectionStore",
    "TranscriptRepository",
    "UserFeedbackRepository",
    "UserInterestRepository",
    "default_storage_dir",
    "default_sqlite_path",
    "resolve_sqlite_path",
]
