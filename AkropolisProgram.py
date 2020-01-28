# Code by Ycreak 2020
import cltk
import pickle 
import gensim
from optparse import OptionParser
import sys
import numpy as np
import pdb
import re

# CLTK Import statements (needs to be sanitized)
from cltk.ir.query import search_corpus
from cltk.corpus.utils.importer import CorpusImporter
from cltk.corpus.readers import get_corpus_reader
from cltk.corpus.greek.alphabet import UPPER_ROUGH_ACUTE

from greek_accentuation.characters import base
from greek_accentuation.characters import add_diacritic
from greek_accentuation.characters import length, strip_length
from greek_accentuation.syllabify import syllabify, display_word
from greek_accentuation.syllabify import is_diphthong
from greek_accentuation.syllabify import ultima, rime, onset_nucleus_coda
from greek_accentuation.syllabify import debreath, rebreath
from greek_accentuation.syllabify import syllable_length, syllable_accent
from greek_accentuation.syllabify import add_necessary_breathing
from greek_accentuation.accentuation import get_accent_type, display_accent_type
from greek_accentuation.accentuation import syllable_add_accent, make_paroxytone
from greek_accentuation.accentuation import possible_accentuations
from greek_accentuation.accentuation import recessive, on_penult

from cltk.corpus.utils.formatter import assemble_tlg_author_filepaths
from cltk.tag.pos import POSTag

from cltk.corpus.greek.tlg.parse_tlg_indices import get_female_authors
from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_index
from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithets
from cltk.corpus.greek.tlg.parse_tlg_indices import select_authors_by_epithet
from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_of_author
from cltk.corpus.greek.tlg.parse_tlg_indices import get_geo_index
from cltk.corpus.greek.tlg.parse_tlg_indices import get_geographies
from cltk.corpus.greek.tlg.parse_tlg_indices import select_authors_by_geo
from cltk.corpus.greek.tlg.parse_tlg_indices import get_geo_of_author
from cltk.corpus.greek.tlg.parse_tlg_indices import get_lists
from cltk.corpus.greek.tlg.parse_tlg_indices import get_id_author
from cltk.corpus.greek.tlg.parse_tlg_indices import select_id_by_name

from cltk.corpus.utils.formatter import assemble_tlg_works_filepaths
from cltk.stem.lemma import LemmaReplacer

from cltk.corpus.greek.tlgu import TLGU
from cltk.vector.word2vec import get_sims

from cltk.corpus.greek import tlg
from cltk.utils import philology

import unicodedata
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
from cltk.tokenize.greek.sentence import SentenceTokenizer

from nltk.tokenize.punkt import PunktLanguageVars
from cltk.stop.greek.stops import STOPS_LIST

from gensim import models
from gensim.models import Word2Vec, KeyedVectors
from cltk.vector.word2vec import get_sims

# Gensim imports
from gensim import corpora
from gensim.models import Word2Vec
from gensim import models

from cltk.corpus.utils.formatter import cltk_normalize

import nltk
from nltk import word_tokenize

from sklearn.decomposition import PCA #Grab PCA functions

import resource

#Visualization imports
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sPCA
from sklearn import manifold #MSD, t-SNE
  
##################
# Class Importer #
##################
from Word2VecVisualiserClass import Word2VecVisualiser

# Command line parser base on the Classification of text documents using sparse 
# features program by Peter Prettenhofer, Olivier Grisel, Mathieu Blondel and Lars Buitinck
# License: BSD 3 clause
op = OptionParser()
op.add_option("--redoLemmatizer",
              action="store_true", dest="redoLemmatizer",
              help="Creates new Lemmatized list in lemmWords.pickle.")
op.add_option("--printResults",
              action="store_true", dest="printResults",
              help="Does what it says on the tin.")
op.add_option("--POSTagger",
              action="store_true", dest="POSTagger",
              help="Reruns the POSTagger.")
op.add_option("--Progress",
              action="store_true", dest="Progress",
              help="Prints progress of the program.")

#########################
# Option Parser Content #
#########################
def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

#########################
# Corpus Importing Part #
#########################
if opts.Progress:
    print('Importing Corpora')

corpus_importer = CorpusImporter('greek')
corpus_importer.import_corpus('greek_word2vec_cltk')
corpus_importer.import_corpus('greek_models_cltk')
corpus_importer.import_corpus('greek_text_perseus')

# Strips the accents from the text for easier searching the text.
# Accepts string with accentuation, returns string without.
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

# Retrieves the Lemmatized List from file.
def retrieveLemmatizedList(file):
    with open(file, 'rb') as f:
        lemmWords = pickle.load(f)
    return lemmWords

# Removes all the numbers and punctuation marks from a given list.
# Returns list without these.
def removeNumbers(list): 
    pattern = '[0-9]|[^\w]'
    list = [re.sub(pattern, '', i) for i in list] 
    return list

# Lemmatizes a given list. Returns a list with lemmatized words.
def lemmatizeList(list):
    tagger = POSTag('greek')

    lemmatizer = LemmaReplacer('greek')
    lemmWords = lemmatizer.lemmatize(list)

    # Remove Stopwords and numbers and lowercases all words.
    lemmWords = [w.lower() for w in lemmWords if not w in STOPS_LIST]
    lemmWords = removeNumbers(lemmWords)

    return lemmWords

# Creates wordlist of the selected work and returns this list of words.
def getWordList(selectedWork):
    reader = get_corpus_reader( corpus_name = 'greek_text_perseus', language = 'greek')
    docs = list(reader.docs())

    reader._fileids = [selectedWork]

    words = list(reader.words())

    return words

# Redo the lemmatizer. Use this when a new corpus of texts has been selected
def redoLemmatizer():
    print('Creating a list of lemmatized words.')

    # Hard coded used text. Should be an array of selected texts given to the function.
    words = getWordList('pausanias__description-of-greece__grc.json')

    if opts.POSTagger:
        lemmWords = POSTagger(words)
        lemmWords = lemmatizeList(lemmWords)
        writeTo = 'lemmWordsPOS.pickle'
    else:
        lemmWords = lemmatizeList(words)
        writeTo = 'lemmWords.pickle'

    with open(writeTo, 'wb') as f:
        pickle.dump(lemmWords, f)

    return lemmWords

# Not used at this moment: tags parts of speech.
def POSTagger(wordList):
    if opts.Progress:
        print('Going for the POSTagger.')

    tagger = POSTag('greek')

    listWithTags = []
    listWithSelected = []

    # Create a list first with all the words with tag
    for word in wordList:
        taggedItem = tagger.tag_tnt(word)
        listWithTags.append(taggedItem)
        print(taggedItem)

    # Select from this list only the words you want
    for entry in listWithTags:
        for word, tag in entry:
            if tag == None:
                break # if tag.startswith("N"):
            elif tag.startswith("N"):
                listWithSelected.append(word)
            elif tag.startswith("V"):
                listWithSelected.append(word)
            elif tag.startswith("Unk"):
                listWithSelected.append(word)

    return listWithSelected

# Generates and returns a word2vec model with given parameters.
def genWord2VecModel(vectorList, size, window):
    return Word2Vec([vectorList], size=size, window=window, min_count=1, workers=4)

# Prints the words most similar to the given words with the generated model.
def word2vecMostSimilar(word, count, word2vec):
    if opts.printResults:
        sim_words = word2vec.wv.most_similar(word,topn=count)
        print('\nCurrent Word: ', word)
        for simWord, similarity in sim_words:
            print(simWord, similarity)
            # print(word2vec.wv[word])

# Visualises the Word2Vec model with the given words.
def visualiseWord2Vec(word2vec, wordhouse):
    # No idea why I used this.
    words_house.reverse()

    words2 = words_house[:]
    words2.reverse()
    Word2VecVisualiser.compare_words_polyline(word2vec,words2)

#################
# MAIN FUNCTION #
#################
if opts.redoLemmatizer:
    lemmWords = redoLemmatizer()
else:
    lemmWords = retrieveLemmatizedList('lemmWords.pickle')

# Word2Vec options
modelDimension = 25
modelWindow = 5
simWords = 10

word2vec = genWord2VecModel(lemmWords, modelDimension, modelWindow)

# Words you want to investigate
words_house = ['ναός','ἱερόν','τέμενος']
for word in words_house:
    word2vecMostSimilar(word, simWords, word2vec)

visualiseWord2Vec(word2vec, words_house)






