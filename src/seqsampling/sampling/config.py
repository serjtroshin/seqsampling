# src/seqsampling/sampling/config.py
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from hydra.core.config_store import ConfigStore

@dataclass
class GenerationConfig:
    """
    A config that is passed to the LLM client for generation parameters.
    """
    n: int = 1           # number of parallel prompts
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 1.0

@dataclass
class SamplingConfig:
    """
    A config for a sampling algorithm.
    """
    k: int = 4           # number of parallel prompts
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    extra_params: Optional[Dict] = None  # penalties, etc.

@dataclass
class ConstrainedGenConfig:
    """
    Configuration for constrained generation modes.
    """
    # generic knobs; backend decides how to use them
    json_mode: bool = False
    json_schema: Optional[dict] = None
    json_schema_name: Optional[str] = None  # optional explicit name for OpenAI-style payloads
    json_schema_strict: Optional[bool] = True  # controls 'strict' flag for JSON schema
    xml_mode: bool = False
    grammar: Optional[str] = None  # e.g. regex/BNF for some backends

    def to_params(self) -> Dict[str, Any]:
        """
        Convert generic constraints into extra_params for the client.
        (This is a 'default' mapping; individual clients can override/extend.)
        """
        params: Dict[str, Any] = {}

        # JSON mode
        if self.json_schema is not None:
            # json_schema is stricter than plain json_mode
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": self._format_json_schema(),
            }
        elif self.json_mode:
            params["response_format"] = {"type": "json_object"}

        # XML-ish mode
        if self.xml_mode:
            # placeholder; actual backends might use different keys
            params["format"] = "xml"

        # Grammar constraints
        if self.grammar:
            params["grammar"] = self.grammar

        return params

    def _format_json_schema(self) -> Dict[str, Any]:
        """
        OpenAI (and compatible) clients expect the schema wrapped as:
        {"name": ..., "schema": {...}, "strict": bool}

        We allow callers to pass either a raw JSON schema (old style) or an
        already-wrapped dict; this normalizes to the expected structure.
        """
        if self.json_schema is None:
            raise ValueError("json_schema must be provided to format it.")

        schema_copy = copy.deepcopy(self.json_schema)
        if not isinstance(schema_copy, dict):
            raise ValueError("json_schema must be a dict when provided.")

        # If user already supplied an OpenAI-style payload, respect it and only
        # fill in missing pieces.
        if "schema" in schema_copy:
            payload: Dict[str, Any] = schema_copy
        else:
            payload = {"schema": schema_copy}

        strict_val = payload.get("strict", self.json_schema_strict)
        if strict_val is not None:
            payload["strict"] = strict_val

        if strict_val:
            payload["schema"] = self._ensure_additional_properties_false(payload["schema"])

        if not payload.get("name"):
            payload["name"] = self.json_schema_name or self._infer_schema_name(payload.get("schema"))

        return payload

    def _ensure_additional_properties_false(self, schema: Any) -> Any:
        """
        Recursively set additionalProperties to False for object schemas when strict.
        OpenAI requires this when using strict JSON schema output.
        """
        if not isinstance(schema, dict):
            return schema

        schema = copy.deepcopy(schema)

        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        # Recurse into common schema containers
        for key in ("properties", "$defs", "definitions"):
            if key in schema and isinstance(schema[key], dict):
                schema[key] = {k: self._ensure_additional_properties_false(v) for k, v in schema[key].items()}

        if "items" in schema:
            items = schema["items"]
            if isinstance(items, list):
                schema["items"] = [self._ensure_additional_properties_false(it) for it in items]
            else:
                schema["items"] = self._ensure_additional_properties_false(items)

        for key in ("oneOf", "anyOf", "allOf"):
            if key in schema and isinstance(schema[key], list):
                schema[key] = [self._ensure_additional_properties_false(v) for v in schema[key]]

        return schema

    @staticmethod
    def _infer_schema_name(schema: Any) -> str:
        if isinstance(schema, dict):
            title = schema.get("title")
            if isinstance(title, str) and title:
                return title
        return "response"

# Register configs
cs = ConfigStore.instance()
cs.store(name="generation_config", node=GenerationConfig)
cs.store(name="sampling_config", node=SamplingConfig)
cs.store(name="constrained_gen_config", node=ConstrainedGenConfig)
