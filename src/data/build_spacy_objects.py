import os
import spacy
import pandas as pd
import csv
import sys
import pickle
from tqdm import tqdm


def build(infile_path, nlp):
    """Builds a dictionary of spacy objects for each instance in the dataset

    Args:
        infile_path (str): path to the data file
        nlp (spacy language model): model for the language

    Returns:
        Nothing. Saves output as pickle files.
    """

    fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset',
                  'target_word', 'native_annots', 'nonnative_annots',
                  'native_complex', 'nonnative_complex', 'gold_label',
                  'gold_prob']

    spacy_objects = {}

    with open(infile_path, encoding='utf-8') as file:

        reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')
        dataset = [sent for sent in reader]

        X = []

        for instance in tqdm(dataset):

            x = nlp(instance['sentence'])
            X.append(x.to_bytes())

        return X


if __name__ == '__main__':
    #TODO write this stuff in the readme, makedir if dir is missing.
    language = sys.argv[1]
    todo_external = None
    if(len(sys.argv)>2):
        todo_external = sys.argv[2]
    save_dir = 'data/interim/' + language
    data_path = 'data/raw/' + language
    file_paths = os.listdir(data_path)

    # Make directory for language
    try:
        print ('making directory')
        os.mkdir(save_dir)
    except:
        pass

    if language == "english":
        nlp = spacy.load('en_core_web_lg')
    elif language == "spanish":
        nlp = spacy.load("es_core_news_md")
    elif language == "german":
        nlp = spacy.load('de_core_news_sm')
    elif language == "french":
        nlp = spacy.load('fr_core_news_md')
    else:
        raise ValueError("Language specified ({}) not supported.".format(language))
    
    if todo_external:
        data_path = "data/raw/external_datasets/"
        file_paths = os.listdir(data_path)
        save_dir = "/cwi/data/interim/external_datasets/"
        os.makedirs(save_dir,exist_ok=True)


    print('Building spaCy objects for {} language'.format(language))

    for file_path in file_paths:
        interim_save_path = save_dir + "/" + file_path.split(".")[0] + "-spacy-objs.pkl"
        if not file_path.endswith(".tsv") or os.path.exists(interim_save_path):
            print(f"Skipping file {file_path}")
            continue
        print("Building spaCy objects for ", file_path)

        X = build(data_path + '/' + file_path, nlp)

        with open(interim_save_path, "wb") as out_file:
            pickle.dump(X, out_file)
