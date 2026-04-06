import pytest

from promptsops.dataset import map_tinyqa_record


def test_map_tinyqa_record_maps_expected_fields() -> None:
    row = {
        "text": "What currency is used in Japan?",
        "label": "Yen",
        "metadata": {"context": "Japan uses the Yen."},
    }

    example = map_tinyqa_record(row)

    assert example.context == "Japan uses the Yen."
    assert example.question == "What currency is used in Japan?"
    assert example.answer == "Yen"


def test_map_tinyqa_record_requires_metadata_context() -> None:
    row = {
        "text": "What currency is used in Japan?",
        "label": "Yen",
        "metadata": {},
    }

    with pytest.raises(ValueError, match="metadata.context"):
        map_tinyqa_record(row)


def test_map_tinyqa_record_requires_text_and_label() -> None:
    bad_question = {"text": "", "label": "Yen", "metadata": {"context": "ctx"}}
    bad_answer = {
        "text": "What currency is used in Japan?",
        "label": "",
        "metadata": {"context": "ctx"},
    }

    with pytest.raises(ValueError, match="'text'"):
        map_tinyqa_record(bad_question)

    with pytest.raises(ValueError, match="'label'"):
        map_tinyqa_record(bad_answer)
