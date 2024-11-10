import json
import os
import re
from collections import defaultdict
import enchant
from nltk.tokenize import word_tokenize


keywords = ["put your code", "insert code here", "your code here", "handling code"]
d = enchant.Dict("en_US")


def check_comment_snippet(comment, snippet):
    comment_words = set(re.findall(r'\b[A-Za-z_]+\b', comment))
    snippet_words = set(re.findall(r'\b[A-Za-z_]+\b', snippet))
    if comment_words == snippet_words:
        return False
    return True


def contains_english(text):
    return bool(re.search(r'[a-zA-Z]', text))


def contains_keywords(text):
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return False


def count_tokens(text):
    return len(word_tokenize(text))


def check_len(text, min_len):
    count = count_tokens(text)
    return count >= min_len


def split_camel_case(string):
    words = re.findall(r'[A-Z][a-z]*|[a-z]+', string)
    return words


def exceptional_case(input_str):
    if (
        input_str.startswith('// L: ')
        or input_str.startswith('// 0.1L\r: ')
        or input_str.startswith('//&i\r ')
        or input_str.startswith('// DONE')
        or input_str.startswith('// get')
        or input_str.startswith('//BOJ1978')
        or input_str.startswith('/// @end')
        or input_str.startswith('//NOI18N')
    ):
        return False
    return True


def check_word(code_summary):
    words = word_tokenize(code_summary)
    new_items = []
    index = 0
    while index < len(words):
        item = words[index]
        if '_' in item:
            parts = item.split('_')
            new_items.extend(parts)
            words.pop(index)
        else:
            index += 1
    words.extend(new_items)
    words = [word for word in words if len(word) >= 1]
    all_words_len = len(words)
    wrong_words_len = 0
    for word in words:
        if not d.check(word):
            new_words = split_camel_case(word)
            for new_word in new_words:
                if not d.check(new_word):
                    wrong_words_len += 1
    return wrong_words_len * 3 <= all_words_len * 2


def find_commented_annotations(java_code):
    pattern = r'//\s*@.*$'
    return re.match(pattern, java_code) is None


def all_filtering_conditions(input_str):
    if (
        contains_english(input_str)
        and find_commented_annotations(input_str)
        and check_word(input_str)
        and exceptional_case(input_str)
    ):
        return True
    return False


def main():
    input_folder = ''
    output_folder = ''
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            with open(input_file, 'r', encoding='utf-8') as infile, open(output_file.replace('json', 'jsonl'), 'w', encoding='utf-8') as outfile:
                for data in json.load(infile):
                    code_summary = data.get('code_summary', '')
                    code_snippets = data.get('code_snippets', [])
                    raw_code = data.get('raw_code', '')
                    if not (
                            all_filtering_conditions(code_summary)
                            and not contains_keywords(raw_code)
                            and check_len(code_summary, 4)
                            and check_comment_snippet(code_summary, raw_code)
                    ):
                        continue
                    place_to_summary = defaultdict(list)
                    place_to_snippet = {}
                    filtered_code_snippets = []
                    sub_id = 0
                    for item in code_snippets:
                        snippet_summary = item['code_summary']
                        snippet_code = item['code_snippet']
                        if (
                                all_filtering_conditions(snippet_summary)
                                and check_len(snippet_summary, 3)
                                and check_comment_snippet(snippet_summary, snippet_code)
                        ):
                            item['sub_id'] = sub_id
                            sub_id += 1
                            filtered_code_snippets.append(item)
                    if len(filtered_code_snippets) == 0:
                        continue
                    code_snippets = filtered_code_snippets
                    for snippet in code_snippets:
                        place = snippet.get('place', '')
                        if isinstance(place, list):
                            place = '_'.join(map(str, place))
                        summary = snippet.get('code_summary', '')
                        if contains_english(summary):
                            place_to_summary[place].append(summary)
                            if place not in place_to_snippet:
                                place_to_snippet[place] = snippet
                    merged_code_snippets = []
                    for place, summaries in place_to_summary.items():
                        if place in place_to_snippet:
                            snippet = place_to_snippet[place]
                            merged_summary = ' '.join(summaries)
                            snippet['code_summary'] = merged_summary
                            merged_code_snippets.append(snippet)
                    data['code_snippets'] = merged_code_snippets
                    if any(snippet['code_snippet'] for snippet in merged_code_snippets):
                        outfile.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main()
