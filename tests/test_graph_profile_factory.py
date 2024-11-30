from conductor.profiles.factory import GraphModelFactoryPipeline, GraphSignatureFactory


def test_graph_model_factory_pipeline():
    triple_types = {
        "SUBSIDIARY": ("COMPANY", "COMPANY"),
        "EMPLOYEE": ("PERSON", "COMPANY"),
    }
    pipeline = GraphModelFactoryPipeline(triple_types)
    pipeline.create_models()
    assert pipeline.relationship_type is not None
    pipeline.create_triple_type_input()
    assert pipeline.triple_type_inputs is not None


def test_graph_signatures_factory():
    triple_types = {
        "SUBSIDIARY": ("COMPANY", "COMPANY"),
        "EMPLOYEE": ("PERSON", "COMPANY"),
    }
    pipeline = GraphModelFactoryPipeline(triple_types)
    pipeline.create_models()
    pipeline.create_triple_type_input()
    factory = GraphSignatureFactory(
        triple_types=pipeline.triple_type_inputs,
        relationship=pipeline.relationship_model,
    )
    factory.create_signatures()
    assert factory.signatures is not None
    map_ = factory.create_signatures_map()
    assert isinstance(map_, dict)
