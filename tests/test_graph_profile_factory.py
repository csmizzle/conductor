from conductor.profiles.factory import GraphModelFactoryPipeline


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
