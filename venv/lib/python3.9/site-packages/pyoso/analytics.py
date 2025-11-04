from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PartitionStatusRange(BaseModel):
    end_key: str = Field(alias="endKey")
    start_key: str = Field(alias="startKey")
    status: str


class PartitionStatus(BaseModel):
    num_failed: int = Field(alias="numFailed")
    num_materialized: int = Field(alias="numMaterialized")
    num_materializing: int = Field(alias="numMaterializing")
    num_partitions: int = Field(alias="numPartitions")
    ranges: list[PartitionStatusRange]


class MaterializationStatus(BaseModel):
    partition_status: Optional[PartitionStatus] = Field(
        default=None, alias="partitionStatus"
    )
    latest_materialization: Optional[datetime] = Field(
        default=None, alias="latestMaterialization"
    )


class DataStatus(BaseModel):
    key: str
    status: MaterializationStatus
    dependencies: list[str]

    def __str__(self):
        status = self.status
        status_parts = []

        if not status.latest_materialization and not status.partition_status:
            status_parts.append("No analytics data")
        if status.latest_materialization:
            status_parts.append(
                f"Last: {status.latest_materialization.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if status.partition_status:
            ps = status.partition_status
            status_parts.append(
                f"Partitions: {ps.num_materialized}/{ps.num_partitions}"
            )
        status_text = f" ({', '.join(status_parts)})" if status_parts else ""
        return f"{self.key}{status_text}"


class DataAnalytics:
    """Container for analytics data with tree-structured display methods."""

    def __init__(self, analytics_data: dict[str, DataStatus]):
        self._analytics_data = analytics_data
        self._root_keys = self._calculate_root_keys()

    def _calculate_root_keys(self) -> list[str]:
        """Calculate root nodes (nodes that are not dependencies of others)."""
        all_dependencies = set()
        for data_status in self._analytics_data.values():
            all_dependencies.update(data_status.dependencies)

        root_keys = [
            k for k in self._analytics_data.keys() if k not in all_dependencies
        ]

        # If no clear roots, return all keys
        if not root_keys:
            root_keys = list(self._analytics_data.keys())

        return root_keys

    def __iter__(self):
        """Iterate over the analytics data keys."""
        return iter(self._analytics_data.items())

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the analytics data."""
        return key in self._analytics_data

    def __len__(self):
        """Get the number of analytics entries."""
        return len(self._analytics_data)

    @property
    def root_keys(self) -> list[str]:
        """Get the root keys (top-level nodes in the dependency tree)."""
        return self._root_keys.copy()

    @property
    def sources(self) -> list[DataStatus]:
        """Get all sources (keys with no dependencies)."""
        return [
            status
            for status in self._analytics_data.values()
            if len(status.dependencies) == 0
        ]

    def get(self, key: str) -> Optional[DataStatus]:
        """Get analytics data for a specific key."""
        return self._analytics_data.get(key)

    def print_tree(self, key: Optional[str] = None):
        """Print analytics data as a tree structure.

        Args:
            key: If provided, print analytics for this specific key and its dependencies.
                 If None, print analytics for all root keys.
        """
        if key is not None:
            self._print_analytics_tree(key, set())
        else:
            # Print all root nodes
            num_roots = len(self._root_keys)
            for i, root_key in enumerate(self._root_keys):
                self._print_analytics_tree(
                    root_key, set(), is_last=(i == num_roots - 1)
                )

    def _print_analytics_tree(
        self, key: str, visited: set, indent: str = "", is_last: bool = True
    ):
        """Recursively print analytics tree for a given key."""
        if key in visited:
            print(f"{indent}├──{key} (circular dependency)")
            return

        visited.add(key)
        data_status = self._analytics_data[key]

        # Determine the prefix for the current node
        prefix = "└── " if is_last else "├── "
        print(f"{indent}{prefix}{data_status}")

        # Print dependencies
        if data_status.dependencies:
            num_dependencies = len(data_status.dependencies)
            for i, dep in enumerate(data_status.dependencies):
                is_last_dep = i == num_dependencies - 1
                dep_indent = indent + ("    " if is_last else "│   ")
                self._print_analytics_tree(
                    dep, visited.copy(), dep_indent, is_last=is_last_dep
                )

        visited.remove(key)
