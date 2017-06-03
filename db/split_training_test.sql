
-- Copyright 2017 Foundation Center. All Rights Reserved.
--
-- Licensed under the Foundation Center Public License, Version 1.0 (the “License”);
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://gis.foundationcenter.org/licenses/LICENSE-1.0.html
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an “AS IS” BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- ==============================================================================

CREATE INDEX doc_id_index
ON agrovoc_autocode.all_data_codes (doc_id);


CREATE TABLE agrovoc_autocode.test_ids
SELECT doc_id
FROM (
	SELECT dist_ids.doc_id, @counter := @counter + 1 AS counter
	FROM (SELECT @counter := 0) AS initvar, (SELECT DISTINCT doc_id FROM agrovoc_autocode.all_data_codes) AS dist_ids
	ORDER BY rand()
) AS a
WHERE counter <= (15 / 100 * @counter);


CREATE UNIQUE INDEX doc_id_index
ON agrovoc_autocode.test_ids (doc_id);


CREATE TABLE agrovoc_autocode.train_ids
SELECT t1.doc_id
FROM (SELECT DISTINCT doc_id FROM agrovoc_autocode.all_data_codes) t1
LEFT OUTER JOIN agrovoc_autocode.test_ids t2 ON t1.doc_id = t2.doc_id
WHERE t2.doc_id IS NULL;


CREATE UNIQUE INDEX doc_id_index
ON agrovoc_autocode.train_ids (doc_id);


-- this is a test to ensure disjoint test and training sets, the count should be 0
SELECT count(*)
FROM agrovoc_autocode.train_ids t1
JOIN agrovoc_autocode.test_ids t2 ON t1.doc_id = t2.doc_id;


CREATE TABLE agrovoc_autocode.train_documents
SELECT t2.doc_id, t2.text, t2.code
FROM agrovoc_autocode.train_ids t1
INNER JOIN agrovoc_autocode.all_data_codes t2 ON t1.doc_id = t2.doc_id;

CREATE INDEX doc_id_index
ON agrovoc_autocode.train_documents (doc_id);

CREATE INDEX code_index
ON agrovoc_autocode.train_documents (code);



CREATE TABLE agrovoc_autocode.test_documents
SELECT t2.doc_id, t2.text, t2.code
FROM agrovoc_autocode.test_ids t1
INNER JOIN agrovoc_autocode.all_data_codes t2 ON t1.doc_id = t2.doc_id;

CREATE INDEX doc_id_index
ON agrovoc_autocode.test_documents (doc_id);

CREATE INDEX code_index
=======
-- Copyright 2017 Foundation Center. All Rights Reserved.
--
-- Licensed under the Foundation Center Public License, Version 1.0 (the “License”);
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://gis.foundationcenter.org/licenses/LICENSE-1.0.html
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an “AS IS” BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- ==============================================================================

CREATE INDEX doc_id_index
ON agrovoc_autocode.all_data_codes (doc_id);


CREATE TABLE agrovoc_autocode.test_ids
SELECT doc_id
FROM (
	SELECT dist_ids.doc_id, @counter := @counter + 1 AS counter
	FROM (SELECT @counter := 0) AS initvar, (SELECT DISTINCT doc_id FROM agrovoc_autocode.all_data_codes) AS dist_ids
	ORDER BY rand()
) AS a
WHERE counter <= (15 / 100 * @counter);


CREATE UNIQUE INDEX doc_id_index
ON agrovoc_autocode.test_ids (doc_id);


CREATE TABLE agrovoc_autocode.train_ids
SELECT t1.doc_id
FROM (SELECT DISTINCT doc_id FROM agrovoc_autocode.all_data_codes) t1
LEFT OUTER JOIN agrovoc_autocode.test_ids t2 ON t1.doc_id = t2.doc_id
WHERE t2.doc_id IS NULL;


CREATE UNIQUE INDEX doc_id_index
ON agrovoc_autocode.train_ids (doc_id);


-- this is a test to ensure disjoint test and training sets, the count should be 0
SELECT count(*)
FROM agrovoc_autocode.train_ids t1
JOIN agrovoc_autocode.test_ids t2 ON t1.doc_id = t2.doc_id;


CREATE TABLE agrovoc_autocode.train_documents
SELECT t2.doc_id, t2.text, t2.code
FROM agrovoc_autocode.train_ids t1
INNER JOIN agrovoc_autocode.all_data_codes t2 ON t1.doc_id = t2.doc_id;

CREATE INDEX doc_id_index
ON agrovoc_autocode.train_documents (doc_id);

CREATE INDEX code_index
ON agrovoc_autocode.train_documents (code);



CREATE TABLE agrovoc_autocode.test_documents
SELECT t2.doc_id, t2.text, t2.code
FROM agrovoc_autocode.test_ids t1
INNER JOIN agrovoc_autocode.all_data_codes t2 ON t1.doc_id = t2.doc_id;

CREATE INDEX doc_id_index
ON agrovoc_autocode.test_documents (doc_id);

CREATE INDEX code_index
ON agrovoc_autocode.test_documents (code);