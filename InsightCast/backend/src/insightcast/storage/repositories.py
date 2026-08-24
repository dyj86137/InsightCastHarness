"""InsightCast 领域模型的 Repository 实现。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, cast

from infra.serialization import deserialize, serialize
from insightcast.domain.enums import CandidateStatus, InterviewStatus
from insightcast.domain.models import (
    CandidateItem,
    DailyBrief,
    IndustryProfile,
    Interview,
    InterviewDecision,
    InterviewSummary,
    Person,
    RunRecord,
    SearchQuery,
    Source,
    Transcript,
    UserFeedback,
    UserInterest,
)
from insightcast.storage.base import (
    InvalidRecordError,
    ModelPredicate,
    ModelT,
    RecordNotFoundError,
    Repository,
)
from insightcast.storage.json_store import JsonCollectionStore, PathLike, default_storage_dir
from insightcast.storage.sqlite_store import (
    SQLiteCollectionStore,
    default_sqlite_path,
    resolve_sqlite_path,
)


class JsonRepository(Repository[ModelT], Generic[ModelT]):
    """把 Pydantic 领域模型保存到 JSON 集合文件。"""

    def __init__(
        self,
        model_type: Type[ModelT],
        collection_name: str,
        root_dir: Optional[PathLike] = None,
        id_field: str = "id",
        store: Optional[Any] = None,
    ) -> None:
        self.model_type = model_type
        self.collection_name = collection_name
        self.id_field = id_field
        self.store = store or JsonCollectionStore(
            collection_name=collection_name,
            root_dir=root_dir,
            id_field=id_field,
        )

    def create(self, model: ModelT) -> ModelT:
        """创建新记录，如果 ID 已存在则失败。"""

        record = self._model_to_record(model)
        saved = self.store.save_record(record, overwrite=False)
        return self._record_to_model(saved)

    def save(self, model: ModelT) -> ModelT:
        """保存记录；如果 ID 已存在则覆盖。"""

        record = self._model_to_record(model)
        saved = self.store.save_record(record, overwrite=True)
        return self._record_to_model(saved)

    def get(self, record_id: str) -> Optional[ModelT]:
        """按 ID 获取领域模型。"""

        record = self.store.get_record(record_id)
        if record is None:
            return None
        return self._record_to_model(record)

    def require(self, record_id: str) -> ModelT:
        """按 ID 获取领域模型，不存在时抛出异常。"""

        model = self.get(record_id)
        if model is None:
            raise RecordNotFoundError(
                f"{self.collection_name} record not found: {record_id}"
            )
        return model

    def list(self, predicate: Optional[ModelPredicate[ModelT]] = None) -> List[ModelT]:
        """列出领域模型，可传入谓词做内存过滤。"""

        models = [self._record_to_model(record) for record in self.store.list_records()]
        if predicate is None:
            return models
        return [model for model in models if predicate(model)]

    def update(self, record_id: str, **updates: Any) -> ModelT:
        """局部更新记录，并重新经过领域模型校验。"""

        current = self.require(record_id)
        payload = self._model_to_record(current)
        payload.update(serialize(updates))
        updated = self._record_to_model(payload)
        return self.save(updated)

    def delete(self, record_id: str) -> bool:
        """删除记录，返回是否真的删除了数据。"""

        return self.store.delete_record(record_id)

    def exists(self, record_id: str) -> bool:
        """判断记录是否存在。"""

        return self.store.exists(record_id)

    def count(self) -> int:
        """返回当前集合中的记录数量。"""

        return self.store.count()

    def find_by(self, **fields: Any) -> List[ModelT]:
        """按字段精确匹配记录。"""

        expected = serialize(fields)
        return self.list(
            lambda model: all(
                serialize(getattr(model, field, None)) == value
                for field, value in expected.items()
            )
        )

    def find_one_by(self, **fields: Any) -> Optional[ModelT]:
        """按字段精确匹配第一条记录。"""

        matches = self.find_by(**fields)
        if not matches:
            return None
        return matches[0]

    def _model_to_record(self, model: ModelT) -> Dict[str, Any]:
        """把领域模型转换成可写入 JSON 的字典。"""

        if not isinstance(model, self.model_type):
            raise InvalidRecordError(
                f"{self.collection_name} expects model: {self.model_type.__name__}"
            )
        return cast(Dict[str, Any], serialize(model))

    def _record_to_model(self, record: Dict[str, Any]) -> ModelT:
        """把原始字典解析成领域模型。"""

        try:
            return cast(ModelT, deserialize(self.model_type, record))
        except Exception as exc:
            raise InvalidRecordError(
                f"{self.collection_name} record cannot be parsed as "
                f"{self.model_type.__name__}"
            ) from exc


class IndustryProfileRepository(JsonRepository[IndustryProfile]):
    """行业配置仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(IndustryProfile, "industry_profiles", root_dir=root_dir, store=store)

    def list_enabled(self) -> List[IndustryProfile]:
        """列出启用中的行业。"""

        return self.list(lambda industry: industry.enabled)


class PersonRepository(JsonRepository[Person]):
    """商业人物仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(Person, "people", root_dir=root_dir, store=store)

    def list_enabled(self) -> List[Person]:
        """列出启用中的人物。"""

        return self.list(lambda person: person.enabled)

    def find_by_name(self, name: str) -> Optional[Person]:
        """按姓名或别名查找人物。"""

        normalized = name.strip().lower()
        for person in self.list():
            names = [value.lower() for value in person.search_names]
            if normalized in names:
                return person
        return None


class SourceRepository(JsonRepository[Source]):
    """内容来源仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(Source, "sources", root_dir=root_dir, store=store)

    def list_enabled(self) -> List[Source]:
        """列出启用中的来源。"""

        return self.list(lambda source: source.enabled)


class UserInterestRepository(JsonRepository[UserInterest]):
    """用户兴趣仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(UserInterest, "user_interests", root_dir=root_dir, store=store)


class SearchQueryRepository(JsonRepository[SearchQuery]):
    """搜索查询仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(SearchQuery, "search_queries", root_dir=root_dir, store=store)


class CandidateRepository(JsonRepository[CandidateItem]):
    """候选内容仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(CandidateItem, "candidates", root_dir=root_dir, store=store)

    def find_by_url(self, url: str) -> Optional[CandidateItem]:
        """按 URL 查找候选内容。"""

        return self._find_by_effective_url(url)

    def find_by_platform_item_id(self, platform_item_id: str) -> Optional[CandidateItem]:
        """按平台内容 ID 查找候选内容。"""

        return self.find_one_by(platform_item_id=platform_item_id)

    def list_by_status(self, status: CandidateStatus) -> List[CandidateItem]:
        """按处理状态列出候选内容。"""

        return self.list(lambda candidate: candidate.status == status)

    def _find_by_effective_url(self, url: str) -> Optional[CandidateItem]:
        """按规范化 URL 或原始 URL 查找候选内容。"""

        target = url.strip()
        for candidate in self.list():
            if candidate.effective_url == target or str(candidate.url) == target:
                return candidate
        return None


class InterviewDecisionRepository(JsonRepository[InterviewDecision]):
    """候选内容访谈判定仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(
            InterviewDecision,
            "interview_decisions",
            root_dir=root_dir,
            id_field="candidate_id",
            store=store,
        )


class InterviewRepository(JsonRepository[Interview]):
    """规范化访谈仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(Interview, "interviews", root_dir=root_dir, store=store)

    def find_by_url(self, url: str) -> Optional[Interview]:
        """按 URL 查找访谈。"""

        target = url.strip()
        for interview in self.list():
            if interview.effective_url == target or str(interview.url) == target:
                return interview
        return None

    def list_by_status(self, status: InterviewStatus) -> List[Interview]:
        """按处理状态列出访谈。"""

        return self.list(lambda interview: interview.status == status)


class TranscriptRepository(JsonRepository[Transcript]):
    """转录文本仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(Transcript, "transcripts", root_dir=root_dir, store=store)

    def find_by_interview_id(self, interview_id: str) -> Optional[Transcript]:
        """按访谈 ID 查找转录文本。"""

        return self.find_one_by(interview_id=interview_id)


class InterviewSummaryRepository(JsonRepository[InterviewSummary]):
    """访谈摘要仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(InterviewSummary, "interview_summaries", root_dir=root_dir, store=store)

    def find_by_interview_id(self, interview_id: str) -> Optional[InterviewSummary]:
        """按访谈 ID 查找摘要。"""

        return self.find_one_by(interview_id=interview_id)


class DailyBriefRepository(JsonRepository[DailyBrief]):
    """每日简报仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(DailyBrief, "daily_briefs", root_dir=root_dir, store=store)

    def find_by_date(self, brief_date: date) -> Optional[DailyBrief]:
        """按日期查找每日简报。"""

        return self.find_one_by(brief_date=brief_date)


class UserFeedbackRepository(JsonRepository[UserFeedback]):
    """用户反馈仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(UserFeedback, "user_feedback", root_dir=root_dir, store=store)


class RunRecordRepository(JsonRepository[RunRecord]):
    """运行记录仓储。"""

    def __init__(self, root_dir: Optional[PathLike] = None, store: Optional[Any] = None) -> None:
        super().__init__(RunRecord, "run_records", root_dir=root_dir, store=store)

    def list_running(self) -> List[RunRecord]:
        """列出仍处于 running 状态的运行记录。"""

        return self.list(lambda record: record.status == "running")

    def finish(
        self,
        record_id: str,
        status: str = "finished",
        error: Optional[str] = None,
        **counts: Any,
    ) -> RunRecord:
        """结束一次运行记录。"""

        updates: Dict[str, Any] = {
            "finished_at": datetime.utcnow(),
            "status": status,
            "error": error,
        }
        updates.update(counts)
        return self.update(record_id, **updates)


class InsightCastRepositories:
    """集中创建 InsightCast V1 的本地仓储。"""

    def __init__(
        self,
        root_dir: Optional[PathLike] = None,
        *,
        backend: str = "json",
        sqlite_path: Optional[PathLike] = None,
    ) -> None:
        self.backend = _normalize_backend(backend)
        if self.backend == "sqlite":
            self.sqlite_path = resolve_sqlite_path(sqlite_path or root_dir)
            self.root_dir = self.sqlite_path.parent
        else:
            self.sqlite_path = None
            self.root_dir = Path(root_dir) if root_dir is not None else default_storage_dir()

        self.industry_profiles = IndustryProfileRepository(
            self.root_dir,
            store=self._store("industry_profiles"),
        )
        self.people = PersonRepository(self.root_dir, store=self._store("people"))
        self.sources = SourceRepository(self.root_dir, store=self._store("sources"))
        self.user_interests = UserInterestRepository(
            self.root_dir,
            store=self._store("user_interests"),
        )
        self.search_queries = SearchQueryRepository(
            self.root_dir,
            store=self._store("search_queries"),
        )
        self.candidates = CandidateRepository(
            self.root_dir,
            store=self._store("candidates"),
        )
        self.interview_decisions = InterviewDecisionRepository(
            self.root_dir,
            store=self._store("interview_decisions", id_field="candidate_id"),
        )
        self.interviews = InterviewRepository(
            self.root_dir,
            store=self._store("interviews"),
        )
        self.transcripts = TranscriptRepository(
            self.root_dir,
            store=self._store("transcripts"),
        )
        self.interview_summaries = InterviewSummaryRepository(
            self.root_dir,
            store=self._store("interview_summaries"),
        )
        self.daily_briefs = DailyBriefRepository(
            self.root_dir,
            store=self._store("daily_briefs"),
        )
        self.user_feedback = UserFeedbackRepository(
            self.root_dir,
            store=self._store("user_feedback"),
        )
        self.run_records = RunRecordRepository(
            self.root_dir,
            store=self._store("run_records"),
        )

    def _store(self, collection_name: str, id_field: str = "id") -> Optional[Any]:
        if self.backend != "sqlite":
            return None
        return SQLiteCollectionStore(
            collection_name=collection_name,
            database_path=self.sqlite_path or default_sqlite_path(),
            id_field=id_field,
        )


def _normalize_backend(value: str) -> str:
    backend = str(value or "json").strip().lower()
    if backend not in ("json", "sqlite"):
        raise ValueError(f"unsupported storage backend: {value}")
    return backend


__all__ = [
    "CandidateRepository",
    "DailyBriefRepository",
    "IndustryProfileRepository",
    "InsightCastRepositories",
    "InterviewDecisionRepository",
    "InterviewRepository",
    "InterviewSummaryRepository",
    "JsonRepository",
    "PersonRepository",
    "RunRecordRepository",
    "SearchQueryRepository",
    "SourceRepository",
    "TranscriptRepository",
    "UserFeedbackRepository",
    "UserInterestRepository",
]
