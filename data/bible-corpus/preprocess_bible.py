import xml.etree.ElementTree as ET
import os
import string
from tqdm import tqdm
import pdb

class BibleParser():

    def __init__(self, filename):
        self.filename = filename
        self.corpus = []

    def parse_bible(self):
        tree = ET.parse(self.filename)
        root = tree.getroot()
        bible_body = root.find('text').find('body')
        for book in bible_body.getchildren():
            for chapter in book.getchildren():
                for verse in tqdm(chapter.getchildren(),f"Processing chapter {chapter.get('id')} from book {book.get('id')}"):
                    self.parse_verse(verse)


    def parse_verse(self,verse):
        id = verse.get('id')
        sentence = verse.text.strip()
        sentence = sentence.translate(str.maketrans("","",string.punctuation))
        words = sentence.split()
        for word in words:
            self.corpus.append(
                (
                    id,
                    sentence,
                    sentence.index(word),
                    sentence.index(word) + len(word),
                    word,
                    10,
                    10,
                    0,
                    0,
                    0,
                    0
                )
            )


    def export_corpus(self):
        with open(self.filename.replace(".xml",".tsv"),'w') as fout:
            for record in self.corpus:
                row = "\t".join(map(str,record)) + "\n"
                fout.write(row)
        print("Corpus successfully exported!")
 

if __name__=='__main__':
    parser = BibleParser('bibles/English.xml')
    parser.parse_bible()
    parser.export_corpus()