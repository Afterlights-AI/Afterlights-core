TOOLS = [
    {
        "type": "function",
        "name": "read_txt",
        "description": "Return the full UTF-8 text of a local txt file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"}
            },
            "required": ["file_path"]
        }
    },
    {
        "type": "function",
        "name": "write_csv",
        "description": "Write the dialogue records to disk as CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "text":    {"type": "string"}
                        },
                        "required": ["speaker", "text"]
                    }
                },
                "output_path": {"type": "string"}
            },
            "required": ["records", "output_path"]
        }
    },
    {
        "type": "function",
        "name": "clean_raw_text",
        "description": "Clean a block of raw text by applying a series of regular-expression rules. Each rule can either be a plain pattern to remove (string) or an object specifying a pattern and an explicit replacement string.",
        "parameters": {
            "type": "object",
            "properties": {
            "regex_rules": {
                "type": "array",
                "description": "A list of regex rules to apply. Each list item may be - \n• a string, treated as a pattern to remove (replaced with a single space), or\n• an object containing a `pattern` string and an optional `replacement` string.",
                "items": {
                "oneOf": [
                    {
                    "type": "string",
                    "description": "Regex pattern to remove (replacement defaults to a single space)."
                    },
                    {
                    "type": "object",
                    "description": "Explicit regex rule with its own replacement.",
                    "properties": {
                        "pattern": {
                        "type": "string",
                        "description": "Regex pattern to match."
                        },
                        "replacement": {
                        "type": "string",
                        "description": "Replacement text for the matched pattern (defaults to an empty string if omitted).",
                        "default": ""
                        }
                    },
                    "required": ["pattern"],
                    "additionalProperties": False
                    }
                ]
                }
            }
            },
            "required": ["regex_rules"],
            "additionalProperties": False
        }
    }

    
]