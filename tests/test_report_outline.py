from conductor.reports.builder.outline import OutlineBuilder, build_outline


def test_outline_builder() -> None:
    section_titles = [
        "Executive Summary",
        "Introduction",
        "Background",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    outline_builder = OutlineBuilder(section_titles=section_titles)
    specification = "The report should be about Thomson Reuters Special Services."
    outline = outline_builder(specification=specification)
    assert isinstance(outline, list)
    assert len(outline) == 7


def test_build_outline() -> None:
    section_titles = [
        "Executive Summary",
        "Introduction",
        "Background",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    specification = "The report should be about Thomson Reuters Special Services."
    outline = build_outline(specification=specification, section_titles=section_titles)
    assert isinstance(outline, list)
    assert len(outline) == 7
