from conductor.utils import clean_string
from tests.constants import TEST_NEWLINE_HTML_DATA


def test_clean_string():
    cleaned_data = clean_string(TEST_NEWLINE_HTML_DATA)
    assert "\n" not in cleaned_data
