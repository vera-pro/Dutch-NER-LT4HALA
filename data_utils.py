from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric, Dataset, DatasetDict

def convert_to_dataset(data, label_map):
    formatted_data = {"tokens": [], "ner_tags": []}
    for sentence in data:
        tokens = [token_data[0] for token_data in sentence if token_data]
        ner_tags = [label_map[token_data[-1]] for token_data in sentence if token_data]
        formatted_data["tokens"].append(tokens)
        formatted_data["ner_tags"].append(ner_tags)
    return Dataset.from_dict(formatted_data)

def get_text(data):
    return " ".join(item[0] for item in data if item)

def get_sentence_splits(data, max_len = 200):
    res = []
    tok_num = 0
    sentences = sent_tokenize(get_text(data))
    cur_sent=[]
    for sent in sentences:
        if len(cur_sent) > max_len:
            res.append(cur_sent)
            cur_sent = []
        for token in sent.split():
            while not data[tok_num]:
                tok_num += 1
#             print(token, data[tok_num],tok_num)
            if data[tok_num][0].startswith(token):
                cur_sent.append(data[tok_num])
                tok_num += 1
                
    if cur_sent:
        res.append(cur_sent)
    return res

def read_conll_file(file_path):
    with open(file_path, "rb") as f:
        content = f.read().decode(errors='replace').strip()
        sentences = content.split("\n\n")
        data = []
        for sentence in sentences:
            if sentence.startswith('-DOCSTART-'):
                continue
            tokens = sentence.split("\n")
            token_data = []
            for token in tokens:
                token_data.append(token.split())
            if token_data:
                data.append(token_data)
    return data

def prepare_data(path):
    data_raw = read_conll_file(path)
    data = []

    for item in tqdm(data_raw):
        data.extend(get_sentence_splits(item))
    return data

