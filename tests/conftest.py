from pathlib import Path

import pytest

from llm_opt_lab.data import Sample, load_samples
from llm_opt_lab.doc_types import DocType, load_doc_types

FIXTURES = Path(__file__).parent / "fixtures"
DATA_DIR = FIXTURES / "data"
DOC_TYPES_DIR = FIXTURES / "doc_types"


@pytest.fixture
def doc_types() -> dict[str, DocType]:
    return load_doc_types(DOC_TYPES_DIR)


@pytest.fixture
def samples() -> list[Sample]:
    return load_samples(DATA_DIR / "test.jsonl")
