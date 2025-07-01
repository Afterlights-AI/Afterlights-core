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
from util import TOOLS
from model_context import ModelContext

### -----------------------------------------------------------
### 1)  The three helper functions the model will call
### -----------------------------------------------------------

def read_txt(file_path: str) -> dict:
    """Return the full text of a TXT file as one UTF-8 string."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return {"raw_text": raw_text[1000:1500]}

def read_full_text(file_path: str) -> dict:
    """Return the full text of a TXT file as one UTF-8 string."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return {"raw_text": raw_text}

def write_csv(raw_text: str, output_path, **args) -> dict:
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
    
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "text"])
        writer.writeheader()
        writer.writerows(records)
    return {"csv_path": str(pathlib.Path(output_path).resolve())}

def write_csv_depr(records, output_path: str) -> dict:
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
        if isinstance(rule, str):
            # If the rule is a string, treat it as a pattern to remove
            pattern = rule
            replacement = " "
        elif isinstance(rule, dict):
            pattern = rule.get("pattern", "")
            replacement = rule.get("replacement", "")
        else:
            continue  # skip invalid rule

        try:
            cleaned = re.sub(
                pattern,
                replacement,
                cleaned,
                flags=re.MULTILINE
            )
        except re.error as e:
            # Skip invalid regex patterns
            continue
    return {"clean_text": cleaned}



# Map tool names to local callables
local_functions = {
    "read_txt":             read_txt,
    "write_csv":            write_csv,
    "clean_raw_text":       clean_raw_text,
}

### -----------------------------------------------------------
### 3)  Minimal driver loop
### -----------------------------------------------------------


def run_agent(pdf_path: str):
    print(pdf_path)
    with open("agent/prompts/dataset_processor.prompt", "r") as f:
        SYSTEM_PRMPT = f.read()
    mc = ModelContext(model_name="gpt-4.1-nano")
    mc.add_system_message(SYSTEM_PRMPT)
    mc.add_user_message(f"Clean this PDF and give me a CSV with columns speaker and text. The path is at {pdf_path}")
    cur_text = read_full_text(pdf_path)["raw_text"]
    piece_text = read_txt(pdf_path)["raw_text"]
    regex_rules = []          # ← add
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("my_logs.log", mode="w", encoding="utf-8")
        ]
    )

    def log_message(role, content):
        logging.info(f"{role.upper()} MESSAGE: {content}")

    def log_tool_call(fn_name, args, result):
        logging.info(f"TOOL CALL: {fn_name}({args}) -> {str(result)}")

    
    while True:
        log_message("user", mc.get_history())
        # log_message("cur_text", cur_text)  # Log first 1000 chars of text
        assistant_msg = mc.call_model_with_tools(TOOLS=TOOLS)
        input(assistant_msg)

        mc.add_function_call(assistant_msg)
        log_message("assistant", assistant_msg)

        # Otherwise, satisfy each tool call and send results back
        call = assistant_msg
        fn_name = call.name
        args = json.loads(call.arguments or "{}")
        log_tool_call(fn_name, args, None)  # Log the tool call            
        if fn_name == "clean_raw_text":
            input(f"output rules were { args}...")
            regex_rules = args['regex_rules']
            args["raw_text"] = piece_text
            piece_text = clean_raw_text(**args)
            mc.add_tool_message(
                tool_call_id=call.call_id,
                tool_result=json.dumps(piece_text)
            )
            cur_text = clean_raw_text(cur_text, regex_rules)["clean_text"]
            continue
            
        if fn_name == "write_csv":
            args["raw_text"] = cur_text
            result = local_functions["write_csv"](**args)
            
            mc.add_tool_message(
                tool_call_id=call.call_id,
                tool_result=json.dumps(result)
            )
            
            break
            
        result = local_functions[fn_name](**args)
        mc.add_tool_message(
            tool_call_id=call.call_id,
            tool_result=json.dumps(result)
        )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python agentic_pdf_to_csv.py path/to/file.pdf")
    run_agent(sys.argv[1])
