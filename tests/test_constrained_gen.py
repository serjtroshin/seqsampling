# tests/test_constrained_gen_config.py
from seqsampling.sampling.config import ConstrainedGenConfig


def test_constrained_gen_config_json_mode():
    cfg = ConstrainedGenConfig(json_mode=True)
    params = cfg.to_params()

    assert "response_format" in params
    assert params["response_format"]["type"] == "json_object"


def test_constrained_gen_config_json_schema_overrides_json_mode():
    schema = {"title": "MySchema", "type": "object", "properties": {"x": {"type": "number"}}}
    cfg = ConstrainedGenConfig(json_mode=True, json_schema=schema)
    params = cfg.to_params()

    assert "response_format" in params
    assert params["response_format"]["type"] == "json_schema"
    json_schema = params["response_format"]["json_schema"]
    assert json_schema["schema"]["properties"] == schema["properties"]
    assert json_schema["schema"]["additionalProperties"] is False
    assert json_schema["name"] == "MySchema"
    assert json_schema["strict"] is True


def test_constrained_gen_config_preserves_existing_schema_payload():
    payload = {"name": "custom", "schema": {"type": "object"}, "strict": False}
    cfg = ConstrainedGenConfig(json_schema=payload, json_schema_strict=True)
    params = cfg.to_params()

    assert params["response_format"]["json_schema"]["name"] == "custom"
    assert params["response_format"]["json_schema"]["schema"] == {"type": "object"}
    # existing strict flag should not be overridden
    assert params["response_format"]["json_schema"]["strict"] is False


def test_constrained_gen_config_adds_additional_properties_when_strict():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    cfg = ConstrainedGenConfig(json_schema=schema, json_schema_strict=True)
    params = cfg.to_params()
    formatted = params["response_format"]["json_schema"]["schema"]

    assert formatted["additionalProperties"] is False
    # nested objects also get additionalProperties: false
    assert formatted["properties"]["x"]["type"] == "string"

def test_constrained_gen_config_xml_and_grammar():
    cfg = ConstrainedGenConfig(xml_mode=True, grammar="S -> 'a' 'b'")
    params = cfg.to_params()

    assert params["format"] == "xml"
    assert params["grammar"] == "S -> 'a' 'b'"
