import yaml, json
import re

def parse_json(json_str):
    # Check if the input is empty or whitespace
    if not json_str or not json_str.strip():
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Empty JSON string received from LLM. Original response: {repr(json_str)}")
        raise ValueError("Empty JSON string received from LLM")

    if "```json" in json_str:
        json_str = json_str.split("```json")[1].rsplit("```", 1)[0].strip()
    else:
        json_str = json_str.strip()

    # Check again after extraction
    if not json_str:
        raise ValueError("Empty JSON string after extracting from code block")

    # Remove or escape invalid control characters in JSON strings
    # This regex finds string values and replaces control characters with escaped versions
    def escape_control_chars(match):
        s = match.group(0)
        # Replace control characters with their escape sequences
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        s = s.replace('\b', '\\b')
        s = s.replace('\f', '\\f')
        # Replace any other control characters (ASCII 0-31) except those already handled
        s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
        return s

    # Apply escaping to string values in the JSON
    # This pattern matches quoted strings, being careful to handle escaped quotes
    json_str = re.sub(r'"(?:[^"\\]|\\.)*"', escape_control_chars, json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Provide more context in the error message
        raise ValueError(f"Failed to parse JSON: {str(e)}\nJSON string (first 500 chars): {json_str[:500]}")from e

def parse_yaml(yaml_str):
    if "```yaml" in yaml_str:
        yaml_str = yaml_str.split("```yaml")[1].rsplit("```", 1)[0].strip()
        return yaml.safe_load(yaml_str)
    else:
        return yaml.safe_load(yaml_str.strip())

# Custom Dumper to enforce double quotes for strings
class DoubleQuoteDumper(yaml.Dumper):
    def represent_str(self, data):
        return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')
DoubleQuoteDumper.add_representer(str, DoubleQuoteDumper.represent_str)

def json_to_yaml_str(json_data):
    # Dump it into a YAML string with the custom Dumper
    yaml_string = yaml.dump(json_data, Dumper=DoubleQuoteDumper, default_flow_style=False, sort_keys=False)
    return yaml_string 