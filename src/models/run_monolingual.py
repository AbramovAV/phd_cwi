"""For running the baseline model

This models runs the baseline model on the datasets of all languages.

"""

import argparse
import nltk
import csv
import os
import numpy as np

from src.data.dataset import Dataset
from src.models.monolingual import MonolingualCWI
from src.models.evaluation import report_binary_score
from collections import Counter
from src.features import file_io


# datasets_per_language = {"english": ["News", "WikiNews", "Wikipedia"],
datasets_per_language = {"english": ["lcp_bible","lcp_biomed","lcp_europarl"],
                         "spanish": ["Spanish"],
                         "german": ["German"],
                         "french": ["French"]}


def run_model(language, dataset_name, evaluation_split, detailed_report, ablate, save_preds, external=None):
    """Trains and tests the CWI model for a particular dataset of a particular language. Reports results.

    Args:
        language: The language of the dataset.
        dataset_name: The name of the dataset (all files should have it).
        evaluation_split: The split of the data to use for evaluating the performance of the model (dev or test).
        detailed_report: Whether to display a detailed report or just overall score.

    """
    score_only = True if ablate else False

    data = Dataset(language, dataset_name)
    #The code below is used for creating unigram probability csv files

        #corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
                                    # ['ignore', 'words', 'ignore', 'ignore', 'ignore'],
                                     #encoding='utf-8')
    # filename = 'europarl-v7.fr-en.fr'
    # file = open(filename, mode='rt', encoding='utf-8')
    # corpus_words = []
    # for line in file:
    #     #print(line)
    #     corpus_words += line.strip(',').strip('.').split()
    #     #print(corpus_words)

    # #corpus_words = corp.words()
    # unigram_counts = Counter(corpus_words)
    # total_words = len(corpus_words)

    # def calc_unigram_prob(unigram_counts, total_words):
    #     u_prob = {} #defaultdict
    #     for word in unigram_counts:
    #         u_prob[word] = unigram_counts[word]/total_words
    #     return u_prob

    # def save_to_file(u_prob,file_name):
    #     w = csv.writer(open(file_name, "w"))
    #     for word, prob in u_prob.items():
    #         w.writerow([word, prob])
    # print('calc unigram prob: ')

    # u_prob = calc_unigram_prob(unigram_counts, total_words)
    # print('saving file')
    # save_to_file(u_prob, 'data/external/french_u_prob.csv')

    baseline = MonolingualCWI(language, ablate,"SVM")
    model_path = f"/cwi//models/{dataset_name}_logistic_regressor.joblib"

    # if os.path.exists(model_path):
    #     baseline.load_model(model_path)
    # else:
    baseline.train(data.train_set())
    #     baseline.export_model(model_path)


    if evaluation_split in ["dev", "both"]:
        if not score_only:
            print("\nResults on Development Data")
        predictions_dev, predictions_dev_proba = baseline.predict(data.dev_set())
        gold_labels_dev = data.dev_set()['gold_label']
        dump_predictions(data.dev_set(),predictions_dev,language, dataset_name+"_dev",predictions_dev_proba,gold_labels_dev)
        print(report_binary_score(gold_labels_dev, predictions_dev, detailed_report, score_only))


    if evaluation_split in ["test", "both"]:
        if not score_only:
            print("\nResults on Test Data")
        predictions_test, predictions_test_proba = baseline.predict(data.test_set())
        gold_labels_test = data.test_set()['gold_label']
        dump_predictions(data.test_set(),predictions_test,language, dataset_name+"_test",predictions_test_proba,gold_labels_test)
        print(report_binary_score(gold_labels_test, predictions_test, detailed_report, score_only))
    if not score_only:
        print()
    if external:
        for dataset in external:
            save_name = dataset_name + "_" + dataset.replace(".tsv",".txt")
            if not dataset.endswith(".tsv") or save_name in external: continue
            print(f"Making predictions over external dataset {dataset}...")
            # external_data = Dataset("english"," ")
            external_test_set = data.external_test_set(dataset)
            predictions_external_test, predictions_external_proba = baseline.predict(external_test_set)
            external_labels_test = data.external_test_set(dataset)['gold_label']
            print(report_binary_score(external_labels_test, predictions_external_test, detailed_report, score_only))
            print(f"Saving predictions over external dataset {dataset}...")
            # dump_external_predictions(external_test_set,predictions_external_test,predictions_external_proba,external_labels_test,dataset,dataset_name)
            

def dump_predictions(test_set,predictions,language, dataset_name,predictions_proba = None,ground_truth = None):
    if not os.path.exists(f"data/predictions/{language}"):
        os.mkdir(f"data/predictions/{language}")
    with open(f"data/predictions/{language}/{dataset_name}"+".tsv",'w') as fout:
        types = ["simple","complex"]
        for idx,type in enumerate(types):
            fout.write(f"Words predicted as {type}" + "\t"*4 + "\n")
            for complex_idx in np.where(predictions==idx)[0]:
                complex_word = test_set.iloc[[complex_idx]]['target_word'].iloc[0]
                sentence = test_set.iloc[[complex_idx]]['sentence'].iloc[0]
                hit_id = test_set.iloc[[complex_idx]]['hit_id'].iloc[0]
                
                # complex_word = complex_word.replace("   ","")
                # sentence = sentence.replace(f"{complex_idx}   ","")
                # hit_id = hit_id.replace(f"{complex_idx}   ","")
                
                proba = predictions_proba[complex_idx][1] if predictions_proba is not None else None
                gt = ground_truth[complex_idx] if ground_truth is not None else None

                data = "\t".join([
                    complex_word,
                    str(predictions[idx]),
                    str(proba),
                    str(gt),
                    sentence,
                    hit_id,
                ]) + "\n"
                fout.write(data)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains and tests the model for all datasets of a language.")
    parser.add_argument('-l', '--language', choices=['english', 'spanish', 'german'], default='english',
                        help="language of the dataset(s).")
    parser.add_argument('-e', '--eval_split', choices=['dev', 'test', 'both'], default='test',
                        help="the split of the data to use for evaluating performance")
    parser.add_argument('-d', '--detailed_report', action='store_true',
                        help="to present a detailed performance report per label.")
    parser.add_argument('-a', '--ablate', help='Runs feature ablation', action='store_true')
    parser.add_argument('-x','--external_datasets',default='',help="contains path to raw external datasets")
    args = parser.parse_args()

    datasets = datasets_per_language[args.language]
    if os.path.exists(args.external_datasets):
        external_datasets = os.listdir(args.external_datasets)
    else:
        external_datasets = None


    for dataset_name in datasets:
        print("\nModel for {} - {}.".format(args.language, dataset_name))

        if not args.ablate:
            run_model(args.language, dataset_name, args.eval_split, args.detailed_report, args.ablate, external_datasets)
        else:
            print('Feature ablation scores:')
            all_feats =[
                'char_tri_sum',
                'char_tri_avg',
                'rare_trigram_count',
                'is_stop',
                'rare_word_count',
                'is_nounphrase',
                'len_tokens_norm',
                'hypernym_count',
                'len_chars_norm',
                'len_tokens',
                'consonant_freq',
                'gr_or_lat',
                'is_capitalised',
                'num_complex_punct',
                'averaged_chars_per_word',
                'sent_length',
                'unigram_prob',
                'char_n_gram_feats',
                #'sent_n_gram_feats',
                'iob_tags',
                'lemma_feats',
                'bag_of_shapes',
                'pos_tag_counts',
                'NER_tag_counts',
                ]

            for feat in all_feats:
                ablate = [feat]
                run_model(args.language, dataset_name, args.eval_split, args.detailed_report, ablate=ablate)
