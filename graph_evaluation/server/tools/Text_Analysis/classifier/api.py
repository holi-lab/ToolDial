import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def job_title_classifier(q: str, cls_data: str=None, want_codes: str=None, output: str=None, detect_langs: str='en', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used to classify a single job title or line of text"
    q: The text to classify
        cls_data: 
effect: provide known classification data relating to the input in order to boost search results

multiple values: repeat parameter for each classification

example: cls_data_ISCO-08
        want_codes: format: classification code, example: ISCO-08

multiple values: repeat parameter for each classification
        output: format: html or empty

default value: empty

effect: output as indended html if set to html, otherwise output as json
        detect_langs: format: ISO 369-1, 2 character language code, example: de, en, fr, …
default value: empty

multiple values: repeat parameter

effect: when the source language is not known, use a combination of python langdetect and the concept graph data to detect the input language. If this is used, the detected languages will be returned as _search_lang_ in the output.

        lang: Format: ISO 369-1, 2 character language code, example: de, en, fr, …


Effect: search in this language, output all data in this language
        
    """
    url = f"https://classifier1.p.rapidapi.com/classifier/classify/"
    querystring = {'q': q, }
    if cls_data:
        querystring['cls_data_'] = cls_data
    if want_codes:
        querystring['want_codes'] = want_codes
    if output:
        querystring['output'] = output
    if detect_langs:
        querystring['detect_langs'] = detect_langs
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "classifier1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def typeahead(q: str, want_codes: str='ISCO-08', num_cls_label_results: str='5', num_results: str='10', output: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint to use typeahead for classification labels and concept labels, using the JANZZ classifier, as an alternative to /concepts/ and /labels/"
    q: Input in the typeahead
        want_codes: only return labels from concepts which are classified with this classification, also include the classification values for each returned concept.

default value: empty

multiple values: repeat parameter
        num_cls_label_results: return N classification labels which match the typeahead input. These will have a concept id (cid) of 0.
        num_results: return N labels
        output: format: html or empty

default value: empty

effect: output as indended html if set to html, otherwise output as json
        
    """
    url = f"https://classifier1.p.rapidapi.com/classifier/typeahead/"
    querystring = {'q': q, }
    if want_codes:
        querystring['want_codes'] = want_codes
    if num_cls_label_results:
        querystring['num_cls_label_results'] = num_cls_label_results
    if num_results:
        querystring['num_results'] = num_results
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "classifier1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

