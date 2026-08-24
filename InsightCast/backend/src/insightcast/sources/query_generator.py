"""从人物、来源和用户兴趣生成平台搜索查询。"""

from __future__ import annotations

from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from insightcast.domain.enums import Industry, SourceType
from insightcast.domain.models import (
    Person,
    SearchQuery,
    Source,
    UserInterest,
    dedupe_non_empty,
)
from insightcast.storage.repositories import (
    InsightCastRepositories,
    SearchQueryRepository,
)


ENGLISH_QUERY_SUFFIXES = (
    ("interview", "person_interview"),
    ("podcast", "person_podcast"),
    ("fireside chat", "person_fireside_chat"),
    ("keynote", "person_keynote"),
)

CHINESE_QUERY_SUFFIXES = (
    ("访谈", "person_interview_zh"),
    ("对话", "person_conversation_zh"),
    ("播客", "person_podcast_zh"),
    ("演讲", "person_speech_zh"),
)


def generate_search_queries(
    people: Sequence[Person],
    sources: Sequence[Source],
    interests: Sequence[UserInterest] = (),
) -> Tuple[SearchQuery, ...]:
    """为启用的人物和来源生成去重后的搜索查询。"""

    active_people = [
        person
        for person in people
        if person.enabled and _person_matches_interests(person, interests)
    ]
    active_sources = [
        source
        for source in sources
        if source.enabled and _source_matches_interests(source, interests)
    ]

    queries: List[SearchQuery] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for source in active_sources:
        for person in active_people:
            if not _person_matches_source(person, source):
                continue
            for language in _source_languages(source):
                for text, template_name in _query_texts(person, language):
                    key = _query_key(source, person, language, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    queries.append(
                        SearchQuery(
                            id=_stable_query_id(source, person, language, text),
                            text=text,
                            person_id=person.id,
                            source_id=source.id,
                            source_type=source.type,
                            language=language,
                            metadata=_query_metadata(source, person, template_name),
                        )
                    )
    return tuple(queries)


def save_search_queries(
    repository: SearchQueryRepository,
    queries: Iterable[SearchQuery],
    *,
    overwrite: bool = True,
) -> int:
    """把生成的搜索查询写入 repository，并返回写入数量。"""

    count = 0
    for query in queries:
        if overwrite:
            repository.save(query)
        else:
            repository.create(query)
        count += 1
    return count


def generate_and_save_search_queries(
    repositories: InsightCastRepositories,
    *,
    overwrite: bool = True,
) -> Tuple[SearchQuery, ...]:
    """从 repositories 读取配置数据，生成并保存搜索查询。"""

    queries = generate_search_queries(
        repositories.people.list_enabled(),
        repositories.sources.list_enabled(),
        repositories.user_interests.list(),
    )
    save_search_queries(
        repositories.search_queries,
        queries,
        overwrite=overwrite,
    )
    return queries


def _query_texts(person: Person, language: str) -> Tuple[Tuple[str, str], ...]:
    normalized_language = _normalize_language(language)
    if normalized_language == "zh":
        return _chinese_query_texts(person)
    return _english_query_texts(person)


def _english_query_texts(person: Person) -> Tuple[Tuple[str, str], ...]:
    values: List[Tuple[str, str]] = []
    for name in _names_for_language(person, "en"):
        for suffix, template_name in ENGLISH_QUERY_SUFFIXES:
            values.append((f"{name} {suffix}", template_name))
    for company in person.companies:
        if _is_ceo_like(person.title):
            values.append((f"{company} CEO interview", "company_ceo_interview"))
    return _dedupe_query_texts(values)


def _chinese_query_texts(person: Person) -> Tuple[Tuple[str, str], ...]:
    values: List[Tuple[str, str]] = []
    for name in _names_for_language(person, "zh"):
        for suffix, template_name in CHINESE_QUERY_SUFFIXES:
            values.append((f"{name} {suffix}", template_name))
    for company in person.companies:
        if _is_ceo_like(person.title):
            values.append((f"{company} CEO 访谈", "company_ceo_interview_zh"))
    return _dedupe_query_texts(values)


def _dedupe_query_texts(values: Iterable[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    seen: Set[str] = set()
    result: List[Tuple[str, str]] = []
    for text, template_name in values:
        normalized = _normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append((" ".join(text.split()), template_name))
    return tuple(result)


def _names_for_language(person: Person, language: str) -> Tuple[str, ...]:
    names = person.search_names
    if language == "en":
        latin_names = [name for name in names if not _contains_cjk(name)]
        return tuple(dedupe_non_empty(latin_names or names))
    if language == "zh":
        cjk_names = [name for name in names if _contains_cjk(name)]
        latin_names = [name for name in names if not _contains_cjk(name)]
        return tuple(dedupe_non_empty(list(cjk_names) + list(latin_names)))
    return names


def _source_languages(source: Source) -> Tuple[str, ...]:
    if source.languages:
        return tuple(dedupe_non_empty(_normalize_language(value) for value in source.languages))
    if source.type == SourceType.BILIBILI:
        return ("zh",)
    return ("en",)


def _person_matches_source(person: Person, source: Source) -> bool:
    if not person.industries or not source.industries:
        return True
    return bool(set(person.industries).intersection(source.industries))


def _person_matches_interests(
    person: Person,
    interests: Sequence[UserInterest],
) -> bool:
    effective_interests = [
        interest for interest in interests if _interest_restricts_people(interest)
    ]
    if not effective_interests:
        return True
    return any(_person_matches_interest(person, interest) for interest in effective_interests)


def _source_matches_interests(
    source: Source,
    interests: Sequence[UserInterest],
) -> bool:
    interest_industries = set()
    for interest in interests:
        interest_industries.update(interest.industries)
    if not interest_industries or not source.industries:
        return True
    return bool(interest_industries.intersection(source.industries))


def _interest_restricts_people(interest: UserInterest) -> bool:
    return bool(interest.people or interest.companies or interest.industries)


def _person_matches_interest(person: Person, interest: UserInterest) -> bool:
    people_targets = {_normalize_text(value) for value in interest.people}
    if people_targets:
        person_names = {_normalize_text(person.id)}
        person_names.update(_normalize_text(value) for value in person.search_names)
        if person_names.intersection(people_targets):
            return True

    company_targets = {_normalize_text(value) for value in interest.companies}
    if company_targets and {
        _normalize_text(value) for value in person.companies
    }.intersection(company_targets):
        return True

    if interest.industries and set(person.industries).intersection(interest.industries):
        return True

    return False


def _query_metadata(
    source: Source,
    person: Person,
    template_name: str,
) -> Dict[str, Any]:
    metadata = {
        "generated_by": "query_generator",
        "query_template": template_name,
        "source_name": source.name,
        "person_name": person.display_name or person.name,
        "source_authority": source.authority,
        "search_scope": _source_search_scope(source),
    }
    if source.platform_id:
        metadata["platform_id"] = source.platform_id
    return metadata


def _source_search_scope(source: Source) -> str:
    value = source.metadata.get("search_scope")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if source.platform_id:
        return "source"
    return "platform"


def _query_key(
    source: Source,
    person: Person,
    language: str,
    text: str,
) -> Tuple[str, str, str, str]:
    return (source.id, person.id, _normalize_language(language), _normalize_text(text))


def _stable_query_id(
    source: Source,
    person: Person,
    language: str,
    text: str,
) -> str:
    raw = "|".join(_query_key(source, person, language, text))
    digest = sha1(raw.encode("utf-8")).hexdigest()[:20]
    return f"query_{digest}"


def _is_ceo_like(title: Optional[str]) -> bool:
    if not title:
        return False
    normalized = title.lower()
    return "ceo" in normalized or "chief executive" in normalized


def _normalize_language(value: str) -> str:
    text = value.strip().lower().replace("_", "-")
    if text.startswith("zh") or text in ("cn", "chinese"):
        return "zh"
    if text.startswith("en") or text == "english":
        return "en"
    return text


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


__all__ = [
    "CHINESE_QUERY_SUFFIXES",
    "ENGLISH_QUERY_SUFFIXES",
    "generate_and_save_search_queries",
    "generate_search_queries",
    "save_search_queries",
]
