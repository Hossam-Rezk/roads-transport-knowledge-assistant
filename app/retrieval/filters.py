"""Translate `SearchFilters` into each store's native filter.

Both functions always include the primary-category restriction, so no search
mode can return a chunk from outside the approved corpus. Filters are executed
by the stores themselves (before ranking), never by prompting the model.
"""
from app.core.models import SearchFilters


def _active_conditions(filters: SearchFilters, primary_category: str) -> dict:
    conditions = {"primary_category": primary_category}
    conditions.update(filters.model_dump(exclude_none=True))
    return conditions


def to_sql_where(filters: SearchFilters, primary_category: str) -> tuple[str, list]:
    """SQL for the FTS5 query, e.g. ("chunks.year = ? AND chunks.status = ?", [2023, "In Force"])."""
    conditions = _active_conditions(filters, primary_category)
    where_sql = " AND ".join(f"chunks.{column} = ?" for column in conditions)
    return where_sql, list(conditions.values())


def to_chroma_where(filters: SearchFilters, primary_category: str) -> dict:
    """Chroma `where` filter. Chroma needs `$and` when there are several conditions."""
    conditions = [{field: value}
                  for field, value in _active_conditions(filters, primary_category).items()]
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
