from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "auto":
        return "auto"
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return ast.literal_eval(value)
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _strip_yaml_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        content = raw.split(" #", 1)[0].rstrip()
        indent = len(content) - len(content.lstrip(" "))
        lines.append((indent, content.strip()))
    return lines


def _parse_simple_yaml(text: str) -> Any:
    """Small YAML subset parser used only when PyYAML is unavailable."""
    lines = _strip_yaml_lines(text)

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        is_list = lines[index][1].startswith("- ")
        if is_list:
            out: list[Any] = []
            while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
                item = lines[index][1][2:].strip()
                index += 1
                if item == "":
                    value, index = parse_block(index, lines[index][0] if index < len(lines) else indent + 2)
                    out.append(value)
                elif ":" in item:
                    key, value_text = item.split(":", 1)
                    entry: dict[str, Any] = {}
                    if value_text.strip():
                        entry[key.strip()] = _parse_scalar(value_text.strip())
                    else:
                        nested, index = parse_block(index, lines[index][0] if index < len(lines) else indent + 2)
                        entry[key.strip()] = nested
                    if index < len(lines) and lines[index][0] > indent:
                        nested_indent = lines[index][0]
                        nested, index = parse_block(index, nested_indent)
                        if isinstance(nested, dict):
                            entry.update(nested)
                    out.append(entry)
                else:
                    out.append(_parse_scalar(item))
            return out, index

        out_dict: dict[str, Any] = {}
        while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
            content = lines[index][1]
            if ":" not in content:
                raise ConfigError(f"Invalid config line: {content}")
            key, value_text = content.split(":", 1)
            key = key.strip()
            value_text = value_text.strip()
            index += 1
            if value_text:
                out_dict[key] = _parse_scalar(value_text)
            else:
                if index < len(lines) and lines[index][0] > indent:
                    nested, index = parse_block(index, lines[index][0])
                    out_dict[key] = nested
                else:
                    out_dict[key] = None
        return out_dict, index

    parsed, consumed = parse_block(0, lines[0][0] if lines else 0)
    if consumed != len(lines):
        raise ConfigError("Could not parse entire YAML file with fallback parser.")
    return parsed


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return expand_env_vars(data)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ConfigError(
                f"Unresolved environment variable in config value '{value}'. "
                "Set it in the shell or override the config path on the command line."
            )
        return expanded
    return value


def dump_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return _dump_simple_yaml(data)


def _dump_simple_yaml(data: Any, indent: int = 0) -> str:
    spaces = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{spaces}{key}:")
                lines.append(_dump_simple_yaml(value, indent + 2))
            else:
                lines.append(f"{spaces}{key}: {value}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{spaces}-")
                lines.append(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{spaces}- {item}")
        return "\n".join(lines)
    return f"{spaces}{data}"


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    config = load_yaml(path)
    required = ["dataset", "training", "experiments"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Config missing required sections: {', '.join(missing)}")
    return config


def find_experiment(config: dict[str, Any], name: str) -> dict[str, Any]:
    for exp in config.get("experiments", []):
        if exp.get("name") == name:
            return exp
    raise ConfigError(f"Experiment '{name}' was not found in config.")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
