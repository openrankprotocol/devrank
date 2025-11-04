"""Provides a dbapi compatible interface for pyoso.

This implementation is far from complete. If you are going to rely on this 
right now things are likely to rapidly change. 
"""

import typing as t

if t.TYPE_CHECKING:
    from .client import Client, QueryResponse


class PyosoDBAPICursor:
    """DBAPI 2.0 compatible cursor protocol for PyOSO."""

    def __init__(self, client: "Client"):
        self._client = client
        self._response: t.Optional["QueryResponse"] = None

    @property
    def description(self) -> t.Optional[list[tuple]]:
        if not self._response:
            return None
        return [
            (col, "", None, None, None, None, None)
            for col in self._response.data.columns
        ]

    @property
    def rowcount(self) -> int:
        if not self._response:
            return -1
        return len(self._response.data.data)

    def execute(self, query: str, params: t.Optional[dict[str, t.Any]] = None) -> t.Any:
        self._response = self._client.query(query)

    def fetchone(self) -> t.Optional[dict[str, t.Any]]:
        if not self._response:
            return None
        return t.cast(dict[str, t.Any], self._response.data.data[0:1])

    def fetchall(self) -> list[dict[str, t.Any]]:
        if not self._response:
            return []
        return t.cast(list[dict[str, t.Any]], self._response.data.data)

    def close(self) -> None:
        return None


class PyosoDBApiConnection:
    """DBAPI 2.0 connection protocol for PyOSO."""

    def __init__(self, client: "Client"):
        self._client = client

    @property
    def dialect(self) -> str:
        return "trino"

    def cursor(self) -> PyosoDBAPICursor:
        return PyosoDBAPICursor(self._client)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None
