from conductor.profiles.factory import create_subclass_with_dynamic_fields, enum_factory
from conductor.flow.models import CitedValue


def test_create_dynamic_value() -> None:
    dynamic_cited_value = create_subclass_with_dynamic_fields(
        model_name="DynamicCitedValue",
        base_class=CitedValue,
        new_fields={
            "value": (bool, None, "The value for the question"),
        },
    )
    # create a new instance of the dynamic subclass to ensure schema works
    dynamic_instance = dynamic_cited_value(
        value=True,
        citations=["https://example.com"],
        faithfulness=5,
        factual_correctness=5,
        confidence=5,
    )
    assert dynamic_instance.value is True


def test_enum_factory() -> None:
    enum = enum_factory("TestEnum", [("A", "A"), ("B", "B")])
    assert enum.A.value == "A"
