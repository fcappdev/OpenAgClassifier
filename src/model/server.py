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
from base.prediction import Predictor
from base.model import TextClassifier
from nltk.data import load

from base.database import MySqlDataBase
from base.graph import run, bfs
from base import config as c

import json
import os
import time
import warnings
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, Response
from flask_cors import cross_origin


app = Flask(__name__)
warnings.simplefilter("ignore", UserWarning)


def _load_to_memory(name, level):
    clf = TextClassifier()
    clf.load(path='model/clf_data/', name=name, in_db=False)
    del clf.doc_ids
    return Predictor(classifier=clf, high_t=c.high_threshold[level], low_t=c.low_threshold[level])


def _get_lookup():
    db = MySqlDataBase(c.db)
    query = """
    SELECT Code, ifnull(ifnull(ifnull(ifnull(ifnull(L7, L6), L5), L4), L3), L2) AS `description`
    FROM (
        SELECT Code, nullif(L7, '') AS L7, nullif(L6, '') AS L6, nullif(L5, '') AS L5
        , nullif(L4, '') AS L4, nullif(L3, '') AS L3, nullif(L2, '') AS L2
        , nullif(L1, '') AS L1
        FROM agrovoc_autocode.agrovoc_terms
        WHERE `Use?` = 'Y'
    ) as a
    """
    db.execute(query)

    d = {}
    for row in db.cursor:
        code = row["Code"].strip()
        description = row["description"].strip()

        d[code] = description
    db.teardown()
    return d


def _validate(js, k):
    return isinstance(js, dict) and k in js


print("[INFO] Loading AGROVOC classifiers")
p1 = _load_to_memory(name='hierarchy_1_76021167-b4ce-463d-bab0-bc7fb044b74b', level=1)
p2 = _load_to_memory(name='hierarchy_2_2fd8b6a0-6786-42ef-9eea-66ea02a1dfdd', level=2)
p3 = _load_to_memory(name='hierarchy_3_2b946288-5eeb-4d35-a1fe-6987c118c3b5', level=3)
p4 = _load_to_memory(name='hierarchy_4_3e787d47-5183-4df2-ba4b-509926f029d3', level=4)

lookup = _get_lookup()
graph = run(MySqlDataBase(c.db))
sentence_detector = load("tokenizers/punkt/english.pickle")


def taxonomy_rollup(results):
    """
    Does the taxonomy rollup using a graph breadth-first-search
    algorithm
    :param results: (list of dictionaries) 
    :return: (list of dictionaries) 
    """

    all_codes = set([r["code"] for r in results])
    to_keep = set()
    node_check = all_codes - to_keep

    for n in node_check:
        to_keep.add(n)
        k = bfs(graph=graph, start=n, to_check=node_check, keep=to_keep)
        to_keep.add(k)

    return [r for r in results if r["code"] in to_keep if r["code"] is not None]


@app.route('/predict', methods=['POST', 'GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def predict():
    """
    Single text predictions
    :return: (JSON)
    """

    j = request.get_json()
    if j is None:
        j = request.args
    if not j:
        j = request.form

    if _validate(j, 'text'):
        st = time.time()
        text = j['text']
        threshold = 0
        chunk = False

        if 'chunk' in j and j['chunk'].lower() == 'true':
            text = [sub for sent in sentence_detector.tokenize(text) for sub in sent.split(';')]
            chunk = True
        if 'threshold' in j and j['threshold'] == 'high':
            threshold = 1

        # get all predictions, for every hierarchy asynchronously
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_results = {executor.submit(func, (text, lookup, threshold)): idx + 1
                              for idx, func in enumerate([p1.predict,
                                                          p2.predict,
                                                          p3.predict,
                                                          p4.predict
                                                          ])}
            for future in as_completed(future_results):
                results.extend(future.result())

        # resolve duplication that arises due to chunking (accept the result with the maximum confidence per class)
        if chunk:
            results_sort = sorted(results, key=lambda x: (x["code"], x["confidence"]))
            grouped = itertools.groupby(results_sort, lambda s: s["code"])
            results = [max(v, key=lambda x: x["confidence"]) for k, v in grouped]

        # add logic to toggle the agrovoc graph roll up on and off
        if 'roll_up' in j and j['roll_up'].lower() == 'false':
            agg = [r for r in results if r["code"] is not None]
        else:
            agg = taxonomy_rollup(results)

        if not agg:
            agg = [{"code": None, "description": None, "confidence": 0.0}]

        agg = sorted(agg, key=lambda s: s["confidence"], reverse=True)
        return Response(response=json.dumps({"success": True, "duration": time.time() - st, "data": agg}, indent=4),
                        status=200,
                        mimetype='application/json')

    return Response(response=json.dumps({"success": False, "status": "Incorrect parameters"}, indent=4),
                    status=404,
                    mimetype='application/json')


if __name__ == '__main__':
    debug = os.environ.get('DEBUG', False)
    port = os.environ.get('PORT', 9091)
    testing = os.environ.get('TESTING', False)
    app.run(host='0.0.0.0', port=port, debug=debug)
