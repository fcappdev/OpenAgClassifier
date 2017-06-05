# -*- coding: utf-8 -*-
"""
# Copyright 2017 Foundation Center. All Rights Reserved.
#
# Licensed under the Foundation Center Public License, Version 1.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://gis.foundationcenter.org/licenses/LICENSE-1.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import numpy as np

import json
import dill
from datetime import datetime
from uuid import uuid4


class TextClassifier:

    def __init__(self):
        self.vectorizer = None
        self.clf = None

        self.doc_ids = None
        self.label2id = None
        self.id2label = None

        self.platt_a = None
        self.platt_b = None
        self.dist_max = None
        self.dist_min = None

    def _get_label_dicts(self, labels):
        """
        Create dictionaries mapping labels to integers 0 to n, in which n is the
        number of unique labels encountered in the given list of labels.
        :param labels: (list)
        """

        sorted_labels = set([l.strip() for ls in labels for l in ls])
        self.label2id = {l.strip(): i for i, l in enumerate(sorted_labels)}
        self.id2label = {i: l.strip() for l, i in self.label2id.items()}

    def _file_save(self, path, filename, platt_a, platt_b, dist_max, dist_min):
        """
        :param path: (string)
        :param filename: (str) 
        :param platt_a: (float)
        :param platt_b: (float)
        :param dist_max: (float)
        :param dist_min: (float)
        """

        with open(path + '{0}_vec.pkl'.format(filename), 'wb') as f:
            dill.dump(self.vectorizer, f)
        with open(path + '{0}_clf.pkl'.format(filename), 'wb') as f:
            dill.dump(self.clf, f)

        with open(path + '{0}.json'.format(filename), 'w') as f:
            d = {'classifier_name': '{0}_clf.pkl'.format(filename),
                 'vectorizer_name': '{0}_vec.pkl'.format(filename),
                 'save_datetime': str(datetime.now()),
                 'parameters': {
                     'PlattA': str(platt_a),
                     'PlattB': str(platt_b),
                     'DistMaximum': str(dist_max.tostring()),
                     'DistMinimum': str(dist_min.tostring()),
                     'DocumentIDs': self.doc_ids,
                     'Labels2IDs': self.label2id}
                 }
            json.dump(json.dumps(d), f, indent=4)

    def _load_from_file(self, path, filename):
        """
        :param path: (string)
        :param filename: (string) 
        """

        with open(path + '{0}.json'.format(filename), 'r') as f:
            metadata = json.loads(json.load(f))

        with open(path + '{0}'.format(metadata['classifier_name']), 'rb') as f:
            self.clf = dill.load(f)
        with open(path + '{0}'.format(metadata['vectorizer_name']), 'rb') as f:
            self.vectorizer = dill.load(f)

        self.platt_a = float(metadata['parameters']['PlattA'])
        self.platt_b = float(metadata['parameters']['PlattB'])
        self.dist_max = np.fromstring(eval(metadata['parameters']['DistMaximum']))
        self.dist_min = np.fromstring(eval(metadata['parameters']['DistMinimum']))
        self.doc_ids = metadata['parameters']['DocumentIDs']
        self.label2id = metadata['parameters']['Labels2IDs']
        self.id2label = {i: l.strip() for l, i in self.label2id.items()}

    def _predict_multi(self, documents, output_positive_score=False):
        """
        Returns label guesses (with probability of accuracy) for each document.
        :param documents: (list)
        :param output_positive_score: (bool, False by default) 
        :return: list of tuples (str, float) of label predictions and associated probabilities
        """

        doc_vectors = self.vectorizer.transform(documents)
        decisions = self.clf.decision_function(doc_vectors)

        a = self.platt_a if self.platt_a is not None else -5.
        b = self.platt_b if self.platt_b is not None else 1.

        pdf = 1. / (1. + np.exp(a * decisions + b))
        assert isinstance(pdf, np.ndarray)

        classes = [str(self.id2label[i]) for i in range(pdf.shape[1])]
        predictions = []
        for ps in pdf:
            if output_positive_score:
                zp = zip(np.array(classes)[decisions[0] > 0].tolist(), [float(x) for x in ps[decisions[0] > 0]])
            else:
                zp = zip(classes, map(lambda x: float(x), ps))
            predictions.append(sorted(zp, reverse=True, key=lambda x: x[1]))
        return predictions

    def label_vectorizer(self, labels):
        """
        Turn a list of labels into an equivalent binarized array of labels.
        :param labels: (list)
        :return: (ndarray)
        """

        label_ids = [[self.label2id[l.strip()] for l in ls] for ls in labels]
        return MultiLabelBinarizer(classes=range(len(self.label2id))).fit_transform(label_ids)

    def train(self, documents, labels, identifiers):
        """
        Fits vectorizer and classifier
        :param documents: (list)
        :param labels: (list)
        :param identifiers: (list)
        """

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, tokenizer=lemma_tokenizer)
        self.clf = OneVsRestClassifier(LinearSVC(random_state=0))

        self._get_label_dicts(labels)
        self.doc_ids = identifiers

        x = self.vectorizer.fit_transform(documents)
        y = self.label_vectorizer(labels)
        self.clf.fit(x, y)

    def predict(self, documents):
        """
        Returns an array of predictions for documents.
        :param documents: (list)
        :return: (ndarray)
        """

        prediction = self._predict_multi(documents)
        return np.array(prediction)[:, 0]

    def grid_predict(self, documents, platt_a, platt_b, low_memory=False):
        """
        Returns label guesses (with probability of accuracy) for each document. This function is only 
        executed when grid searches for the parameters of Platt's posterior probability bootstrapping
        algorithm are being performed.  Otherwise predict_multi is run.
        :param documents: (list)
        :param platt_a: Platt parameter A (float) 
        :param platt_b: Platt parameter B (float)
        :param low_memory: (bool)
        :return: ndarray if low_memory is True, list if low_memory is False
        """

        decisions = self.decision_function(documents)

        if low_memory:
            pdf = np.exp(platt_a * decisions + platt_b).astype(np.float16)
            pdf += 1.
            return 1. / pdf

        pdf = 1. / (1. + np.exp(platt_a * decisions + platt_b))
        assert isinstance(pdf, np.ndarray)

        classes = [self.id2label[i] for i in range(pdf.shape[1])]
        predictions_bulk = []
        for ps in pdf:
            prediction = zip(classes, ps)
            prediction = sorted(prediction, reverse=True, key=lambda s: s[1])
            predictions_bulk.append(prediction)
        return predictions_bulk

    def decision_function(self, documents):
        """
        Returns the decision function values
        :param documents: (list)
        :return: (ndarray)
        """

        doc_vectors = self.vectorizer.transform(documents)
        return self.clf.decision_function(doc_vectors)

    def save(self, path, name, platt_a, platt_b, dist_max, dist_min, in_db=False):
        """
        :param path: (string)
        :param name: (str) 
        :param platt_a: (float)
        :param platt_b: (float)
        :param dist_max: (float)
        :param dist_min: (float)
        :param in_db: (bool)
        """

        file_name = '{0}_{1}'.format(name, uuid4())
        if not in_db:
            self._file_save(path, file_name, platt_a, platt_b, dist_max, dist_min)
            return file_name
        else:
            raise NotImplementedError

    def load(self, path, name, in_db=False):
        """
        :param path: (string)
        :param name: (str) 
        :param in_db: (bool)
        """

        if not in_db:
            self._load_from_file(path, name)
        else:
            raise NotImplementedError


lemmatizer = WordNetLemmatizer()
rTokenizer = RegexpTokenizer('\w+|\$[\d\.]+')
english_stops = stopwords.words('english')


def lemmatize_word(word):
    """
    Lemmatizes word, doesn't distinguish tags, changes word to lower case.
    :param word: (str)
    :return: (str)
    """

    return lemmatizer.lemmatize(lemmatizer.lemmatize(word.lower(), pos='v'), pos='n')


def lemma_tokenizer(text):
    """
    Tokenizes text, lemmatizes words in text and returns the non-stop words (pre-lemma).
    :param text: (str)
    :return: (tuple)
    """

    return [lemmatize_word(word.lower()).strip()
            for word in rTokenizer.tokenize(re.sub(r'á', ' ', text.replace("'s", "")))
            if(word.lower()) not in english_stops]
