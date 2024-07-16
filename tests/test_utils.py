from conductor.utils import gibberish_detector_chunks, is_gibberish


def test_gibberish_detector() -> None:
    """
    Test the gibberish_detector_chunks function.
    """
    text = "This is a test"
    result = gibberish_detector_chunks(text)
    assert isinstance(result, dict)
    assert "noise" not in result
    assert "word_salad" not in result
    assert "mild_gibberish" not in result
    assert "clean" in result


def test_non_gibberish_200k_tokens() -> None:
    """
    Test the gibberish function with 200k words.
    """
    text = " ".join(["word"] * 50000)
    result = is_gibberish(text)
    assert isinstance(result, bool)
    assert not result


def test_gibberish_200k_tokens() -> None:
    """
    Test the gibberish function with 200k words.
    """
    text = " ".join(["asbg"] * 50000)
    result = is_gibberish(text)
    assert isinstance(result, bool)
    assert result
