from math import ceil, floor
from tqdm import tqdm

COMPLEXITY_THRESHOLD = 0.5

def parse_tsv(mode='train'):
    dataset = {}
    with open(f"/cwi/data/raw/lcp/lcp_train/lcp_single_{mode}.tsv") as fin:
        fin.readline()
        for line in fin:
            id, corpus, sentence, token, complexity = line.split('\t')
            if corpus not in dataset: dataset[corpus] = []
            complexity = float(complexity)
            dataset[corpus].append([id,sentence,token,complexity])
    return dataset


def create_dataset(data):
    for i in tqdm(range(len(data))):
        id, sentence, token, complexity = data[i]
        start_pos = sentence.index(token)
        end_pos = start_pos + len(token)
        complexity = round((round(complexity,2) // 0.05) * 0.05,2)
        binary_tag = int(complexity>=0.5)
        data[i] = [
            id,
            sentence.replace("\"",""),
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


def export_tsv(dataset, corpus, mode = 'train'):
    with open(f"/cwi/data/raw/english/lcp_{corpus}_{mode.capitalize()}.tsv",'w') as fout:
        for record in dataset:
            fout.write("\t".join(record) + "\n")

if __name__=='__main__':
    for mode in ['train','test']:
        dataset = parse_tsv(mode)
        for corpus in dataset:
            create_dataset(dataset[corpus])
            export_tsv(dataset[corpus], corpus, mode)