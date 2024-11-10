import json
from sentence_transformers import SentenceTransformer


def Sentence_Bert_encode(input_path, output_path):
    res = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            raw_code = data['raw_code']
            code_embeddings = model.encode(raw_code).tolist()
            data['code_embeddings'] = code_embeddings
            res.append(data)
            if len(res) % 1000 == 0:
                print(len(res))
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    input_path = ""
    output_path = ""
    model = SentenceTransformer('./models/Sentence-Bert')
    Sentence_Bert_encode(input_path, output_path)
