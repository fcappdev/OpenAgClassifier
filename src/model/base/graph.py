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
from copy import deepcopy


def run(db):
    """
    Runs the entire graph construction algorithm
    :param db: (MySQL database class object)
    :return: (dict)
    """

    g = {}
    _build_graph(db, g, ['hierarchy_1', 'hierarchy_2', 'hierarchy_3', 'hierarchy_4'], 'hierarchy_1')
    _build_graph(db, g, ['hierarchy_2', 'hierarchy_3', 'hierarchy_4'], 'hierarchy_2')
    _build_graph(db, g, ['hierarchy_3', 'hierarchy_4'], 'hierarchy_3')
    db.teardown()

    _non_solitary(g)
    return g


def bfs(graph, start, to_check, keep):
    """
    Graph breadth-first-search to do the final predicted
    AGROVOC code rollup. Updates final set in-place.
    :param graph: (dict)
    :param start: (str)
    :param to_check: (set)
    :param keep: final set of nodes, updated in-place (set)
    """

    explored, queue = set(), []
    queue.append(start)
    explored.add(start)

    while queue:
        v = queue.pop(0)
        if v not in graph:
            continue
        for e in graph[v]:
            if e in explored:
                continue
            if e in to_check:
                if v in keep:
                    keep.remove(v)  # remove parent
                keep.add(e)  # add child node
            queue.append(e)
            explored.add(e)


def _build_graph(db, graph, columns, start_column):
    """
    Builds the directed graph of the AGROVOC taxonomy terms
    :param db: (MySQL database class object)
    :param graph: (dict)
    :param columns: (list)
    :param start_column: (str)
    """

    query = """
    SELECT {0}
    FROM agrovoc_autocode.codes_hierarchy
    WHERE {1} IS NOT NULL;
    """.format(','.join(columns), start_column)

    db.execute(query)
    for row in db.cursor:
        node = row[start_column].strip()
        if node not in graph:
            graph[node] = set()
        prev_full = False
        for col in columns[1:]:
            if row[col] is not None:
                if not prev_full:
                    graph[node].add(row[col].strip())
                    prev_full = True
                else:
                    break


def _non_solitary(graph):
    """
    Remove singular nodes
    :param graph: (dict)
    """

    g_copy = deepcopy(graph)
    for n in g_copy:
        if not graph[n]:
            graph.pop(n)
