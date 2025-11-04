# ruff: noqa: F401

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyoso")
except PackageNotFoundError:
    __version__ = "unknown"

from .analytics import DataAnalytics
from .client import Client, ClientConfig, QueryResponse
from .exceptions import HTTPError, OsoError
