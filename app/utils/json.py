"""JSON utilities."""

import ast
import json as std_json
import re

import orjson


class JsonObjectParseError(ValueError):
    """Raised when text does not contain a parseable JSON object."""


def dumps(data: object) -> str:
    """Serialize data to JSON text."""
    return orjson.dumps(data).decode("utf-8")


def loads(data: str) -> object:
    """Deserialize JSON text."""
    return orjson.loads(data)


def loads_json_object(raw_text: str) -> dict:
    """Extract and parse a JSON object from possibly wrapped model output."""
    candidate = extract_json_object(raw_text)
    try:
        data = std_json.loads(candidate)
    except std_json.JSONDecodeError:
        data = _loads_repaired_object(candidate)

    if not isinstance(data, dict):
        raise JsonObjectParseError("Parsed JSON value is not an object")
    return data


def extract_json_object(raw_text: str) -> str:
    """Return the first balanced JSON object from plain text or a code fence."""
    text = raw_text.strip()
    if not text:
        raise JsonObjectParseError("JSON object text is empty")

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    candidate = _find_first_balanced_object(text)
    if candidate is None:
        raise JsonObjectParseError("JSON object was not found")
    return candidate


def _loads_repaired_object(candidate: str) -> object:
    repaired = _repair_common_json(candidate)
    if repaired != candidate:
        try:
            return std_json.loads(repaired)
        except std_json.JSONDecodeError:
            pass

    try:
        return ast.literal_eval(candidate)
    except (SyntaxError, ValueError) as exc:
        raise JsonObjectParseError("JSON object is not parseable") from exc


def _repair_common_json(candidate: str) -> str:
    text = candidate.strip().lstrip("\ufeff")
    text = text.translate(str.maketrans({"“": '"', "”": '"', "„": '"'}))
    return re.sub(r",\s*([}\]])", r"\1", text)


def _find_first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        start = text.find("{", start + 1)

    return None
