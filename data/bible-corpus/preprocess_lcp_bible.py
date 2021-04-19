from math import ceil, floor
from tqdm import tqdm

COMPLEXITY_THRESHOLD = 0.5

def parse_tsv():
    dataset = []
    with open("lcp_train/lcp_single_train.tsv") as fin:
        fin.readline()
        for line in fin:
            id, corpus, sentence, token, complexity = line.split('\t')
            if corpus != 'bible':
                break
            complexity = float(complexity)
            dataset.append([id,sentence,token,complexity])
    return dataset


def create_bible_dataset(data):
    for i in tqdm(range(len(data))):
        id, sentence, token, complexity = data[i]
        start_pos = sentence.index(token)
        end_pos = start_pos + len(token)
        complexity = (round(complexity,2) // 0.05) * 0.05
        binary_tag = int(complexity>=0.5)
        data[i] = [
            id,
            sentence,
            str(start_pos),
            str(end_pos),
            token,
            "10",
            "10",
            str(ceil(complexity * 10)),
            str(floor(complexity * 10)),
            str(binary_tag),
            str(complexity)
        ]


def export_bible_tsv(bible_dataset):
    with open("lcp_train/lcp_bible_train.tsv",'w') as fout:
        for record in bible_dataset:
            fout.write("\t".join(record) + "\n")

if __name__=='__main__':
    bible_dataset = parse_tsv()
    create_bible_dataset(bible_dataset)
    export_bible_tsv(bible_dataset)