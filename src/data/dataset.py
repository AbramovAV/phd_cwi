"""Dataset Reader

This module contains the class(es) and functions to read the datasets.

"""
import csv
import pandas as pd
import spacy
import pickle
from spacy.tokens import Doc


class Dataset(object):
    """
    Utility class to easily load the datasets for training, development and testing.
    """

    def __init__(self, language, dataset_name):
        """Defines the basic properties of the dataset reader.

        Args:
            language: The language of the dataset.
            dataset_name: The name of the dataset (all files should have it).

        """
        self._language = language
        self._dataset_name = dataset_name
        self.translate = 'F'

        # TODO: Maybe the paths should be passed as parameters or read from a configuration file.
        self._trainset_path = "data/raw/{}/{}_Train.tsv".format(language.lower(), dataset_name)
        self._devset_path = "data/raw/{}/{}_Dev.tsv".format(language.lower(), dataset_name)
        self._testset_path = "data/raw/{}/{}_Test.tsv".format(language.lower(), dataset_name)

        self._trainset = None
        self._devset = None
        self._testset = None
        self._external_testset = None

        """spaCy object handling"""
        if self._language == "english":
            self.nlp = spacy.load('en_core_web_lg')
        elif self._language == "spanish":
            self.nlp = spacy.load("es_core_news_md")
        elif self._language == "german":
            self.nlp = spacy.load('de_core_news_sm')
        elif self._language == "french":
            self.nlp = spacy.load('fr_core_news_md')

        self._trainset_spacy_path = "data/interim/{}/{}_Train-spacy-objs.pkl".format(
            language.lower(), dataset_name)
        self._devset_spacy_path = "data/interim/{}/{}_Dev-spacy-objs.pkl".format(
            language.lower(), dataset_name)
        self._testset_spacy_path = "data/interim/{}/{}_Test-spacy-objs.pkl".format(
            language.lower(), dataset_name)

    def train_set(self):
        """list. Getter method for the training set. """
        if self._trainset is None:  # loads the data to memory once and when requested.
            trainset_raw = self.read_dataset(self._trainset_path)
            trainset_spacy = self.read_spacy_pickle(self._trainset_spacy_path)
            if trainset_raw is None and trainset_spacy is None:
                # This is for languages we never see (French)
                self._trainset = None
            else:
                self._trainset = pd.concat([trainset_raw, trainset_spacy], axis=1)

                self._trainset['language'] = self._language
                self._trainset['dataset_name'] = self._dataset_name

        return self._trainset

    def dev_set(self):
        """list. Getter method for the development set. """
        if self._devset is None:  # loads the data to memory once and when requested.
            devset_raw = self.read_dataset(self._devset_path)
            devset_spacy = self.read_spacy_pickle(self._devset_spacy_path)
            if devset_raw is None and devset_spacy is None:
                # This is for languages we never see (French)
                self._devset = None
            else:
                self._devset = pd.concat([devset_raw, devset_spacy], axis=1)

                self._devset['language'] = self._language
                self._devset['dataset_name'] = self._dataset_name

        return self._devset

    def test_set(self):
        """list. Getter method for the test set. """
        
        if self._testset is None:  # loads the data to memory once and when requested.
            
            if self.translate == 'F':
                testset_raw = self.read_dataset(self._testset_path)
            else:
                testset_raw = pickle.load(open( "data/processed/translated_frenchdf.p", "rb" ))
                
            testset_spacy = self.read_spacy_pickle(self._testset_spacy_path)
            self._testset = pd.concat([testset_raw, testset_spacy], axis=1)
            self._testset['language'] = self._language
            self._testset['dataset_name'] = self._dataset_name

        return self._testset

    def external_test_set(self,path):

        # if self._external_testset is None:
        testset_raw = self.read_dataset(f"data/raw/external_datasets/{path}")
        spacy_path = path.replace(".tsv","-spacy-objs.pkl")
        testset_spacy = self.read_spacy_pickle(f"data/interim/external_datasets/{spacy_path}")
        self._external_testset = pd.concat([testset_raw, testset_spacy], axis=1)
        self._external_testset['language'] = self._language
        self._external_testset['dataset_name'] = self._dataset_name

        return self._external_testset


    def read_dataset(self, file_path):
        """Read the dataset file.

        Args:
            file_path (str): The path of the dataset file. The file should follow the structure specified in the
                    2018 CWI Shared Task.

        Returns:
            list. A list of dictionaries that contain the information of each sentence in the dataset.

        """
        try:
            with open(file_path, encoding="utf-8") as file:
                fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                              'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']

                dataset = pd.read_csv(file, names=fieldnames, sep="\t")

        except FileNotFoundError:
            print("File {} not found.".format(file_path))
            dataset = None

        return dataset

    def read_spacy_pickle(self, file_path):
        """Read the pickled spacy objects

        Args:
            file_path (str): Path of the pickled spacy objects file

        Returns:
            pandas DataFrame. A single column of the spacy docs.

        """

        vocab = self.nlp.vocab

        try:
            file = open(file_path, "rb")
            # putting the spacy doc in a single-item list to avoid pandas splitting it up
            spacy_objects = [[Doc(vocab).from_bytes(x)] for x in pickle.load(file)]
            file.close()

            spacy_objects_dataset = pd.DataFrame(spacy_objects, columns=["spacy"])
            return spacy_objects_dataset

        except FileNotFoundError:
            print('spaCy pickle file for {} does not exist. No spaCy objects will be included.'.format(
                self._dataset_name))
            return None
