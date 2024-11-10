import re
import pandas as pd
import json
from tqdm import tqdm
import random

def get_next_non_empty_line(source_code, line_number):
    lines = source_code.splitlines()

    if line_number < 0 or line_number > len(lines)-1:
        return "EOF"

    for i in range(line_number, len(lines)):
        if lines[i].strip() and "// " not in lines[i]:
            return lines[i]

    return "EOF"

def get_lines_range(source_code, start_line, end_line):
    lines = source_code.splitlines()

    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return "EOF"

    arr = []
    for line in lines[start_line - 1:end_line-1]:
        if line != "\n" and "// " not in line:
            arr.append(line)
    if len(arr) != 0:
        return "\n".join(arr)
    else:
        return "EOF"

def split_into_last_line_and_others(code_snippet):
    lines = code_snippet.splitlines()

    if not lines:
        return "", ""

    last_line = lines[-1] if lines else ""
    other_lines = "\n".join(lines[:-1])

    return other_lines, last_line


def generate_line_pairs(code_snippet):
    lines = code_snippet.splitlines()
    pairs = []

    for i in range(1, len(lines)):
        previous_lines = "\n".join(lines[:i])
        current_line = lines[i]

        pairs.append((previous_lines, current_line))
    return pairs


if __name__ == '__main__':
    dataset_file = 'manual_dataset.jsonl'
    process_file1 = 'mutitask_manual_dataset_type1.jsonl'
    process_file2 = 'mutitask_manual_dataset_type2.jsonl'
    process_file3 = 'mutitask_manual_dataset_type3.jsonl'


    with open(dataset_file, "r") as file_in, open(process_file1, "w") as file_out1, open(process_file2, "w") as file_out2, open(process_file3, "w") as file_out3:
        for line in tqdm(file_in):
            data = json.loads(line)
            raw_code = data["raw_code"]
            cur = 2
            length = len(raw_code.splitlines())
            for code_snippet in data["code_snippets"]:
                snippet = code_snippet["code_snippet"]
                start, end = code_snippet["place"]
                if start > cur:
                    snippet_mid = get_lines_range(raw_code, cur, start)
                    next_line = get_next_non_empty_line(raw_code, start-1)
                    if snippet_mid != "EOF" and snippet_mid.strip() != "":
                        file_out1.write(json.dumps({'snippet1': snippet_mid.strip(), 'snippet2': next_line.strip(), 'Label1': 1, 'Label2': 0}) + '\n')
                cur = end + 1

                if start != end:
                    pairs = generate_line_pairs(snippet)
                    for other_lines, last_line in pairs:
                        if other_lines.strip() != "" and last_line.strip() != "":
                            file_out2.write(json.dumps({'snippet1': other_lines.strip(), 'snippet2': last_line.strip(), 'Label1': 0, 'Label2': 0}) + '\n')

                next_line = get_next_non_empty_line(raw_code, end)

                if next_line != "EOF":
                    file_out3.write(json.dumps({'snippet1': snippet.strip(), 'snippet2': next_line.strip(), 'Label1': 1, 'Label2': 1}) + '\n')