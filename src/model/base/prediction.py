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
import numpy as np
from datetime import datetime


class Predictor:

    def __init__(self, classifier, high_t=0, low_t=0):
        self.classifier = classifier
        self.high_t = high_t
        self.low_t = low_t

    def _make_selections(self, prediction):
        """
        Makes multi-label selections based on learned Bayes parameters
        :param prediction: (ndarray)
        :return: (tuple of ndarrays)
        """

        code, score = map(lambda x: np.array(x), zip(*prediction))
        score = score.astype(np.float32)

        threshold = (self.classifier.dist_max + 2 * self.classifier.dist_min) / 3.
        predicted_labels = code[score > threshold]
        predicted_scores = score[score > threshold]

        if len(predicted_labels) == 0:
            predicted_labels, predicted_scores = np.array([None]), np.array([0.0])
        return predicted_labels, predicted_scores

    def predict(self, args):
        """
        Runs multi-class/multi-label prediction algorithm
        :param args: contains two objects in the tuple of the form 
        (text (list or str), lookup (dict), threshold (int)) (tuple)
        :return: (dict)
        """

        if len(args) == 3:
            documents, lookup, threshold = args
            if not isinstance(documents, list):
                if not isinstance(documents, str):
                    raise TypeError
                documents = [documents]
            if not isinstance(lookup, dict):
                raise TypeError
            if not isinstance(threshold, int):
                raise TypeError
            t = self.high_t if threshold == 1 else self.low_t

        else:
            raise SyntaxError("The args must be of the form (documents, lookup dict, threshold bool)")

        platt_a, platt_b = self.classifier.platt_a, self.classifier.platt_b
        predictions = self.classifier.grid_predict(documents, platt_a, platt_b, low_memory=False)
        del documents

        d = []
        for idx, p in enumerate(predictions):
            predicted_labels, predicted_scores = self._make_selections(p)
            for label, score in zip(predicted_labels, predicted_scores.astype(float)):
                w_score = (score - t) / (1. - t)
                if score >= t:
                    if lookup is None:
                        d.append({"code": label, "confidence": w_score})
                    else:
                        d.append({"code": label, "confidence": w_score,
                                  "description": lookup[label].lower() if label is not None else "None"})
        if len(d) > 1:
            return [item for item in d if item["code"] is not None]
        return d

    def predict_bulk(self, documents, ids, level=1, tags=None):
        """
        Runs multi-class/multi-label prediction algorithm
        :param documents: (list)
        :param ids: (list)
        :param level: (int)
        :param tags: misc. tags for the tags column on bulk predictions (str or None)
        :return: (dict)
        """

        if not isinstance(documents, list):
            documents = [documents]

        platt_a, platt_b = self.classifier.platt_a, self.classifier.platt_b
        predictions = self.classifier.grid_predict(documents, platt_a, platt_b, low_memory=False)
        del documents

        d = []
        for p, idx in zip(predictions, ids):
            predicted_labels, predicted_scores = self._make_selections(p)
            for label, score in zip(predicted_labels, predicted_scores.astype(str)):
                d.append((idx, label, score, level, tags, str(datetime.now())))
        return d
