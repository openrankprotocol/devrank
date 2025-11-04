from typing import Callable, Optional

import pandas as pd
import requests
from pydantic import BaseModel, Field
from pyoso.exceptions import OsoHTTPError


class SemanticColumn(BaseModel):
    name: str
    type: str
    description: Optional[str]


class SemanticRelationship(BaseModel):
    source_column: str = Field(alias="sourceColumn")
    target_table: str = Field(alias="targetTable")
    target_column: str = Field(alias="targetColumn")


class SemanticTableResponse(BaseModel):
    name: str
    description: Optional[str]
    columns: list[SemanticColumn] = Field(default_factory=list)
    relationships: list[SemanticRelationship] = Field(default_factory=list)


def query_dynamic_models(base_url: str, api_key: str) -> list[SemanticTableResponse]:
    headers = {
        "Content-Type": "application/json",
    }
    headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.get(
            f"{base_url}connector",
            headers=headers,
        )
        response.raise_for_status()

        response_data = response.json()

        tables = [
            SemanticTableResponse.model_validate(table) for table in response_data
        ]
        return tables
    except requests.HTTPError as e:
        raise OsoHTTPError(e, response=e.response) from None


def create_registry(
    base_url: str, api_key: str, to_pandas_fn: Callable[[str], pd.DataFrame]
):
    from oso_semantic import Dimension, Model
    from oso_semantic import QueryBuilder as InnerQueryBuilder
    from oso_semantic import (
        Registry,
        Relationship,
        RelationshipType,
        register_oso_models,
    )

    class QueryBuilder(InnerQueryBuilder):
        def __init__(self, registry: Registry):
            super().__init__(registry)

        def to_pandas(self):
            sql = self.build()
            return to_pandas_fn(sql.sql(dialect="trino"))

    registry = Registry(QueryBuilder)

    register_oso_models(registry)

    tables = query_dynamic_models(base_url, api_key)
    for table in tables:
        model_name = table.name.split(".")[-1]
        registry.register(
            Model(
                name=model_name,
                description=table.description or "",
                table=table.name,
                dimensions=[
                    Dimension(
                        name=column.name,
                        column_name=column.name,
                        description=column.description or "",
                    )
                    for column in table.columns
                ],
                relationships=[
                    Relationship(
                        name=f"{relationship.source_column}->{relationship.target_table.split('.')[-1]}.{relationship.target_column}",
                        type=RelationshipType.MANY_TO_ONE,
                        source_foreign_key=relationship.source_column,
                        ref_model=relationship.target_table.split(".")[-1],
                        ref_key=relationship.target_column,
                    )
                    for relationship in table.relationships
                ],
            )
        )

    return registry
