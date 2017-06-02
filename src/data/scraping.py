#!/usr/bin/env python3
"""
# Copyright 2017 Foundation Center. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
import dryscrape
from urllib.parse import urlencode
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

import re


class FAOScraper:

    def __init__(self, start_xvfb=False, url=None):
        self.base = url if url is not None else "http://agris.fao.org/agris-search/searchIndex.action?"
        self.search_prefix = "http://agris.fao.org/agris-search/"
        self.xml_prefix = "http://agris.fao.org/agris-search/export!endNote.do?"

        self.start_index_search = 0
        self.num_pages = None
        self.current_page = 1

        self.type_string = ''
        self.sort_field = 'Relevance'
        self.sort_order = 'Descending'
        self.enable_field = 'Disable'
        self.params = ['query',
                       'startIndexSearch',
                       'agrovocString',
                       'typeString',
                       'sortField',
                       'sortOrder',
                       'enableField']

        if start_xvfb:
            dryscrape.start_xvfb()
        self.session = dryscrape.Session()

    def _read(self, url=None):
        if url is None:
            url = self.base
        self.session.visit(url)
        return BeautifulSoup(self.session.body(), 'lxml')

    @staticmethod
    def _download(url, filename):
        urlretrieve(url, filename)

    def _attributes(self, q='*', ag_str=''):
        args = [q,
                self.start_index_search,
                ag_str, self.type_string,
                self.sort_field,
                self.sort_order,
                self.enable_field]

        params = list(zip(self.params, args))
        return self.base + urlencode(params)

    def _get_num_pages(self, page):
        item = page.select(".pagination-row .right nav .pagination .pag-gotolink #goToPage")
        n_pages = item[0]['onkeypress'].split(',')[1]
        self.num_pages = int(n_pages)
        if self.num_pages == 0:
            self.num_pages = 1

    def _paginate(self):
        self.start_index_search += 10
        self.current_page += 1

    def get_search_results(self, ag_str, q='*'):
        """
        Runs searches on the AGRIS engine, handles pagination and sends results
        links to other functions
        :param ag_str: AGROVOC term (str)
        :param q: query (str)
        :return: result URLs (list)
        """

        url = self._attributes(q=q, ag_str=ag_str)
        page = self._read(url)

        if self.num_pages is None:
            self._get_num_pages(page)

        results = page.select(".search-results .result-item .inner .inner h3 a")
        self._paginate()
        return [item['href'] for item in results if item is not None]

    def get_item(self, address):
        """
        Scrapes the title, abstract and AGROVOC codes from the given address
        :param address: (str)
        :return: title, abstract, terms (list) (tuple)
        """

        url = self.search_prefix + address
        page = self._read(url)

        title = page.select_one("article div h1")
        title = title.string.strip() if title is not None else ""
        title = re.sub(r"\s+", " ", title)

        abstract = page.select("article .abstract .row")
        abstract = [re.sub(r"\s+", " ", item.string.strip()) for item in abstract if item.string is not None]
        abstract = abstract[0] if abstract else ""

        terms = page.select(".mashup-blocks div .agrovoc_keywords div ul")
        terms = [item['href'].split('/')[-1] for block in terms for item in block.select("li a")]

        return title, abstract, terms

    def get_xml(self, address, file_location):
        """
        Downloads the XML for the particular record in the address
        :param address: (str)
        :param file_location: (str)
        """

        item_id = address.split("=")[-1]
        address = "arn=" + item_id
        url = self.xml_prefix + address
        self._download(url, file_location + item_id + '.xml')
        return item_id
