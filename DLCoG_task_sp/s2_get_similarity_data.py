import json
import re


def code_add_snippet_comment(code, snippet_comment, place):
    code_lines = code.splitlines()
    new_code = ''
    for i, code_line in enumerate(code_lines, start=1):
        if i == place[0]:
            new_code += snippet_comment + '\n'
        new_code += code_line + '\n'
    return new_code


def remove_comments(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = remove_blank_lines(code)
    return code


def remove_comments_save_blank(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code


def remove_blank_lines(code):
    lines = code.splitlines()
    non_blank_lines = [line for line in lines if line.strip() != '']
    cleaned_code = '\n'.join(non_blank_lines)
    return cleaned_code


def list_to_str(array):
    s = ''
    for a in array:
        s += a
    s = re.sub(r'\s+', '', s)
    return s


def find_sub_text_lines_range(text, sub_text):
    code_lines = text.splitlines()
    sub_code_lines = sub_text.splitlines()
    num_sub_lines = len(sub_code_lines)
    for i in range(len(code_lines)):
        if list_to_str(sub_code_lines) == list_to_str(code_lines[i : i + num_sub_lines]):
            return i + 1, i + num_sub_lines
    return None, None


def process_raw_code(code, code_snippets):
    code = remove_comments(code)
    comment_lines = []
    for i, item in enumerate(code_snippets):
        sub_id = int(item['sub_id']) + 1
        place = item['place']
        code_snippet = remove_comments(item['code_snippet'])
        mask = "// <code_comment{}> range[{}, {}]".format(str(sub_id), str(place[0]), str(place[1]))
        line1, line2 = find_sub_text_lines_range(code, code_snippet)
        if line1 is not None:
            comment_lines.append(line1)
            code = code_add_snippet_comment(code, mask, [line1, line2])
    for i, item in enumerate(code_snippets):
        sub_id = int(item['sub_id']) + 1
        code_snippet = remove_comments_save_blank(item['code_snippet'])
        place = item['place']
        line1, line2 = find_sub_text_lines_range(remove_comments_save_blank(code), code_snippet)
        if line1 is not None:
            mask = "// <code_comment{}> range[{}, {}]".format(str(sub_id), str(place[0]), str(place[1]))
            new_mask = "// <code_comment{}> range[{}, {}]".format(str(sub_id), str(line1), str(line2))
            code = code.replace(mask, new_mask, 1)
    return code


def find_and_replace_multiline_string(code, target_string):
    start_index = code.find(target_string)
    if start_index == -1:
        return code, None, None
    end_index = start_index + len(target_string)
    start_line = code.count('\n', 0, start_index) + 1
    end_line = code.count('\n', 0, end_index) + 1
    lines = code.split('\n')
    extracted_code = '\n'.join(lines[start_line - 1:end_line])
    return extracted_code, start_line, end_line


def extract_code_by_line_range(code, start_line, end_line):
    lines = code.split('\n')
    extracted_code = '\n'.join(lines[start_line - 1:end_line])
    return extracted_code


def process_code_snippet(data):
    for i, item in enumerate(data['code_snippets']):
        code_snippet = item['code_snippet']
        place = item['place']
        if place[0] is None or place[1] is None:
            _, place[0], place[1] = find_and_replace_multiline_string(data['raw_code'], code_snippet)
        if re.sub(r'\s+', '', code_snippet) != re.sub(r's\+', '',
                                                      extract_code_by_line_range(data['raw_code'], place[0], place[1])):
            code_snippet, place[0], place[1] = find_and_replace_multiline_string(data['raw_code'], code_snippet)
            data['code_snippets'][i]['code_snippet'] = code_snippet
            data['code_snippets'][i]['place'] = place
    return data


if __name__ == '__main__':
    samples_cosine_path = ''
    train_file_path = ''
    test_file_path = ''
    output_path = ''
    with open(test_file_path, 'r', encoding='utf-8') as file:
        test_dataset = [json.loads(line) for line in file]
    with open(train_file_path, 'r', encoding='utf-8') as file:
        train_dataset = [json.loads(line) for line in file]

    samples_cosine = []
    with open(samples_cosine_path, 'r', encoding='utf-8') as file:
        for line in file:
            s = json.loads(line)
            for _, value in s.items():
                samples_cosine.append(value)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('')
    for i, test_data in enumerate(test_dataset):
        print(i)
        test_data = process_code_snippet(test_data)
        res = {"test": {}}
        res['test'].update(test_data)
        res['test']['processed_raw_code'] = process_raw_code(test_data['raw_code'], test_data['code_snippets'])
        res['test']['processed_raw_code_without_snippet'] = remove_comments(test_data['raw_code'])
        sample_cosine = samples_cosine[i][:10]
        train_samples_cosine = []
        for j, _index in enumerate(sample_cosine, start=1):
            train_data = process_code_snippet(train_dataset[_index])
            train_data: dict
            train_samples_cosine.append(train_data)
            r = {
                "processed_raw_code": process_raw_code(train_data['raw_code'], train_data['code_snippets']),
                "processed_raw_code_without_snippet": remove_comments(train_data['raw_code']),
                "method_comment": train_data['code_summary']
            }
            for k, item in enumerate(test_data['code_snippets']):
                r['snippet_comment' + str(item['sub_id'] + 1)] = item['code_summary']
            res['example' + str(j)] = r
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res) + '\n')
