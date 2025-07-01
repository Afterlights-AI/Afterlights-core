"""
agentic_pdf_to_csv.py
---------------------
Example “function-calling” agent that:

1. extracts text from a PDF,
2. parses out dialogue into {speaker, text} records,
3. writes them to a CSV.

You run it like:   python agentic_pdf_to_csv.py the_courage_to_be_disliked.pdf
(Requires OPENAI_API_KEY in your environment.)
"""

import os, sys, json, re, csv, pathlib
import openai
from PyPDF2 import PdfReader
import logging


### -----------------------------------------------------------
### 1)  The three helper functions the model will call
### -----------------------------------------------------------

def read_txt(file_path: str) -> dict:
    """Return the full text of a TXT file as one UTF-8 string."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return {"raw_text": raw_text[1000:1500]}

def parse_dialogue(raw_text: str) -> dict:
    """
    Simple heuristic parser.
    A speaker token is a line starting with a name (letters, spaces, or hyphens), followed by a colon.
    Everything until the next speaker token belongs to that speaker.
    """
    speaker_re = re.compile(r"^([A-Za-z][A-Za-z\s\-]+):\s*(.*)$", re.MULTILINE)

    records = []
    current_speaker = None
    buffer = []

    def flush():
        if current_speaker is not None:
            records.append(
                {"speaker": current_speaker.strip(),
                 "text": " ".join(line.strip() for line in buffer).strip()}
            )
            buffer.clear()

    for line in raw_text.splitlines():
        m = speaker_re.match(line)
        if m:
            # starting a new block
            flush()
            current_speaker = m.group(1)
            buffer.append(m.group(2))
        else:
            buffer.append(line)
    flush()

    return {"records": records}

def write_csv(records, output_path: str) -> dict:
    """Write records to a CSV file; returns the absolute path."""
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "text"])
        writer.writeheader()
        writer.writerows(records)
    return {"csv_path": str(pathlib.Path(output_path).resolve())}

### -----------------------------------------------------------
### 2)  Function specs to give the LLM
### -----------------------------------------------------------

def clean_raw_text(raw_text: str, regex_rules: list[dict]) -> dict:
    cleaned = raw_text
    for rule in regex_rules:
        cleaned = re.sub(
            rule["pattern"],      # pattern from the agent
            rule["replacement"],  # replacement text
            cleaned,
            flags=re.MULTILINE
        )
    return {"clean_text": cleaned}

tools = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "clean_raw_text",
    #         "description": "Apply a list of regex rules to clean the raw text (remove page numbers, headers, etc).",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "raw_text": {"type": "string"},
    #                 "regex_rules": {
    #                     "type": "array",
    #                     "items": {
    #                         "type": "object",
    #                         "properties": {
    #                             "pattern": {"type": "string"},
    #                             "replacement": {"type": "string"},
    #                             "note": {"type": "string"}
    #                         },
    #                         "required": ["pattern", "replacement"]
    #                     }
    #                 }
    #             },
    #             "required": ["raw_text", "regex_rules"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "read_txt",
            "description": "Return the full UTF-8 text of a local txt file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"]
            }
        }
    },
    # {
    #     "type": "function",
    #         "function": {
    #         "name": "parse_dialogue",
    #         "description": "Turn raw book text into an array of {speaker, text}.",
    #         "parameters": {
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_regexes",
            "description": ("Given the raw book text, propose a minimal list of "
                            "regex substitutions that will remove page numbers, "
                            "headers/footers, hyphen line-wraps, etc. "
                            "Return an array of objects "
                            "{pattern:str, replacement:str, note:str}. "
                            "The patterns must be valid Python 're' syntax."),
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string"}
                },
                "required": ["raw_text"]
            }
        }
    }
    
]

# Map tool names to local callables
local_functions = {
    "read_txt": read_txt,
    "write_csv":             write_csv,
}

### -----------------------------------------------------------
### 3)  Minimal driver loop
### -----------------------------------------------------------

def clean_raw_text(raw_text: str, regex_rules: list[dict]) -> dict:
    cleaned = raw_text
    for rule in regex_rules:
        cleaned = re.sub(
            rule["pattern"],      # pattern from the agent
            rule["replacement"],  # replacement text
            cleaned,
            flags=re.MULTILINE
        )
    return {"clean_text": cleaned}

def run_agent(pdf_path: str):
    print(pdf_path)
    with open("agent/prompts/dataset_processor.prompt", "r") as f:
        SYSTEM_PRMPT = f.read()
    messages = [
        {"role": "system",
         "content": SYSTEM_PRMPT},
        {"role": "user",
         "content": f"Clean this PDF and give me a CSV with columns speaker and text. The path is at {pdf_path}"}
    ]
    
    cur_text = read_txt(pdf_path)["raw_text"]
    
    regex_rules = []          # ← add
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("my_logs.log", mode="a", encoding="utf-8")
        ]
    )

    def log_message(role, content):
        logging.info(f"{role.upper()} MESSAGE: {content}")

    def log_tool_call(fn_name, args, result):
        logging.info(f"TOOL CALL: {fn_name}({args}) -> {str(result)}")

    
    while True:
        input()
        log_message("user", messages)
        
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # let the model decide
        )
        assistant_msg = response.choices[0].message
        log_message("assistant", assistant_msg.content)
        messages.append(assistant_msg)

        # If the model decided to return normal content, we’re done.
        if assistant_msg.tool_calls is None:
            print(assistant_msg.content)
            break

        # Otherwise, satisfy each tool call and send results back
        for call in assistant_msg.tool_calls:
            
            fn_name = call.function.name
            log_message("tool", f"Calling {fn_name} with args: {call.function.arguments}")
            args = json.loads(call.function.arguments or "{}")
            print(f" running {fn_name}({args})")
            if fn_name == "generate_regexes":
                regex_rules = json.loads(args["raw_text"])  # assistant puts the list here
            # Echo back as a 'tool' response so it’s in the transcript
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({"regex_rules_saved": len(regex_rules)})
                })
                cur_text = clean_raw_text(cur_text["raw_text"], regex_rules)["clean_text"]
                continue

        # —— for clean_raw_text we inject the regex list ——
            if fn_name == "clean_raw_text":
                args["regex_rules"] = regex_rules   # pass the list in
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })
                continue
            
            if fn_name == "parse_dialogue":
                args["raw_text"] = cur_text
                result = local_functions[fn_name](**args)
                
               
            if fn_name == "write_csv":
                args["records"] = cur_text
                args["output_path"] = "output.csv"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })
                break
            
            result = local_functions[fn_name](**args)
            messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python agentic_pdf_to_csv.py path/to/file.pdf")
    run_agent(sys.argv[1])
