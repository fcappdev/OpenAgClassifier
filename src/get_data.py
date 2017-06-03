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
from data.scraping import FAOScraper
from model.base.database import MySqlDataBase
from model.base import config as c

from xml.etree import ElementTree as Et
import os
from time import sleep


conn = c.db


def get_doc_ids():
    """
    Gets all doc_ids in the db if re-running the scraping.
    :return: AGRIS document IDs (set)
    """
    db = MySqlDataBase(conn)
    query = "SELECT DISTINCT doc_id FROM agrovoc_autocode.agris_data"

    doc_ids = set()
    db.execute(query)
    for row in db.cursor:
        doc_ids.add(row['doc_id'].strip())

    query = "SELECT search_term FROM agrovoc_autocode.agris_data GROUP BY search_term"
    terms = set()
    db.execute(query)
    for row in db.cursor:
        terms.add(row['search_term'].strip())

    db.teardown()
    return doc_ids, terms


def get_codes():
    """
    Gets all AGROVOC terms of interest
    :return: AGROVOC codes set, AGROVOC descriptions set (tuple)
    """
    db = MySqlDataBase(conn)
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

    ag_codes = set()
    descriptions = []
    db.execute(query)
    for row in db.cursor:
        ag_codes.add(row['Code'].strip())
        descriptions.append(row['description'].lower().strip())

    db.teardown()
    return ag_codes, descriptions


def scrape(scraper, agrovoc_term, all_codes):
    """
    Runs AGRIS search, scrapes all results from the JavaScript
    :param scraper: scraping class (class)
    :param agrovoc_term: (str)
    :param all_codes: (set)
    :return: results list, page number, total number of pages (tuple)
    """
    page = scraper.get_search_results(ag_str=agrovoc_term)
    page_num, total_pages = s.current_page, s.num_pages

    results = []
    for _, link in enumerate(page):
        title, abstract, codes = scraper.get_item(link)
        codes = set(codes)
        codes = codes & all_codes  # intersection

        if len(codes) > 0:
            text_to_store = title + " - " + abstract
            results.append((link, text_to_store[:4000], ";".join(list(codes)), str(page_num - 1), agrovoc_term))

    return results, page_num, total_pages


def scrape_from_xml(scraper, agrovoc_term, ids=None):
    """
    Runs search on AGRIS, downloads XMLs for every returned record
    and kicks off processing
    :param scraper: scraping class (class)
    :param agrovoc_term: (str)
    :param ids: document IDs already captured (set)
    :return: results list, page number, total number of pages (tuple)
    """
    page = scraper.get_search_results(ag_str=agrovoc_term)
    page_num, total_pages = scraper.current_page, scraper.num_pages
    results = []

    for _, link in enumerate(page):

        if link.split("=")[-1] in ids:
            continue

        item_id = scraper.get_xml(link, 'data/xml/')
        meta_data = process_xml('data/xml/' + item_id + '.xml', page_num, agrovoc_term)
        meta_data = list(meta_data)

        results.append(meta_data[0])

    return results


def process_xml(filename, page_num, agrovoc_term):
    """
    Extracts the title, abstract and AGROVOC codes from 
    temporary XML files stored locally.
    :param filename: (str)
    :param page_num: (int)
    :param agrovoc_term: (str)
    :return: doc_id, text, codes, page number, AGROVOC term (tuple)
    """
    tree = Et.parse(filename)
    root = tree.getroot()

    doc_id = filename.split('/')[-1].split('.')[0]

    for block in root.findall('records'):
        for record in block.findall('record'):
            title = " - ".join([elem.text for elem in record.find('titles')])

            abstracts = [elem.text for elem in record.findall('abstract')]
            abstract = " - ".join(abstracts) if abstracts else ''

            codes = "|".join([elem.text.lower() for elem in record.find('keywords')])
            text_to_store = title + " - " + abstract

            yield (doc_id, text_to_store[:4000], codes, str(page_num - 1), agrovoc_term)
    os.remove(filename)


if __name__ == '__main__':
    codes_to_use, desc = get_codes()
    all_docs, search_terms = get_doc_ids()

    insert = """
    INSERT INTO agrovoc_autocode.agris_data (doc_id, text, codes, page, search_term)
    VALUES (%s, %s, %s, %s, %s)
    """

    start_xvfb = True
    for idx, ag_desc in enumerate(desc):
        s = FAOScraper(start_xvfb=start_xvfb)
        start_xvfb = False

        if ag_desc in search_terms:
            continue

        iterate = True
        while iterate:
            try:
                # pn is page number, tp is total number of pages
                items_to_insert = scrape_from_xml(s, ag_desc, ids=all_docs)

                if items_to_insert:
                    print("Inserting {0} records".format(len(items_to_insert)))
                    database = MySqlDataBase(conn)
                    database.execute_many(insert, items_to_insert)
                    database.teardown()
                    sleep(2)
            except IndexError:
                print("[INFO] IndexError on search term {0}, moving on...".format(ag_desc))
                iterate = False
                s.session.reset()
                continue
            except Exception as ex:
                print("[INFO] An error occurred: {0}".format(ex))
                s.current_page += 1
                s.start_index_search += 10

            if s.current_page - 1 >= s.num_pages:
                s.session.reset()
                iterate = False
