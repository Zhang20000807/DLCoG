import random
random.seed(1)
import concurrent.futures
import requests
from tqdm import tqdm
import json
import re



def ask_gpt(messages):
    try:
        # put your ask LLM code here
        return
    except Exception as e:
        return ask_gpt(messages)

pre_prompt = "You are a Java programmer, please generate code snippet comments for the <code snippet> in the code based on the example, and generate a method comment describing the function of the entire code:\n"

def process_data_with_gpt(data, in_context_num, raw_code_type):
    p = pre_prompt
    for i in range(1, in_context_num + 1):
        p += "\n### Example Code{}:\n".format(i)
        p += data[f"example{i}"][raw_code_type] + "\n"
        p += "### Code Comment:\n"
        sippet_idx = 1
        if raw_code_type == "processed_raw_code":
            while f"snippet_comment{sippet_idx}" in data[f"example{i}"]:
                p += "<snippet comment{}>:".format(sippet_idx) + "{" + data[f"example{i}"][
                    f"snippet_comment{sippet_idx}"] + "}" + "\n"
                sippet_idx += 1
        p += "<method comment>:" + "{" + data[f"example{i}"]["method_comment"] + "}" + "\n"

    p += "\n### Test Code:\n"
    p += data["test"][raw_code_type] + "\n"
    p += "### Code Comment:\n"
    response = ask_gpt([{"role": "user", "content": p}])

    res = {}
    res["response"] = response
    res["result"] = {"snippet": [], "method": {}}
    snippet_len = len(data["test"]["code_snippets"])

    for i in range(1, snippet_len + 1):
        tag_base = "<snippet comment"
        number = i
        tag = f"{tag_base}{number}>"
        pattern = rf"{re.escape(tag)}:\{{(.*?)\}}"
        match = re.search(pattern, response, re.DOTALL)
        temp = {}
        if not match:
            temp["gpt_res"] = "ERROR"
            temp["ground_truth"] = data["test"]["code_snippets"][i - 1]["code_summary"]
        else:
            temp["gpt_res"] = match.group(1).strip()
            temp["ground_truth"] = data["test"]["code_snippets"][i - 1]["code_summary"]
        res["result"]["snippet"].append(temp)

    pattern = r'<method comment>:\{(.*?)\}'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        res["result"]["method"]["gpt_res"] = "ERROR"
        res["result"]["method"]["ground_truth"] = data["test"]["code_summary"]
    else:
        res["result"]["method"]["gpt_res"] = match.group(1).strip()
        res["result"]["method"]["ground_truth"] = data["test"]["code_summary"]

    data["res"] = res
    return data

if __name__ == '__main__':
    in_context_num = 10 # 1/5/10
    raw_code_type = "processed_raw_code" # "processed_raw_code" or "processed_raw_code_without_snippet"

    train_path = ''
    output_path = f"ans_{raw_code_type}_{in_context_num}.jsonl"
    with open(train_path, "r") as fin, open(output_path, "w") as fout:
        data_lines = []
        for line in tqdm(fin):
            try:
                data = json.loads(line)
                data_lines.append(data)
            except json.decoder.JSONDecodeError as e:
                continue
        print(f"Finished with {len(data_lines)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(process_data_with_gpt, data, in_context_num, raw_code_type) for data in tqdm(data_lines)]

            pbar = tqdm(total=len(data_lines))
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                json.dump(res, fout)
                fout.write('\n')
                pbar.update(1)
            pbar.close()




