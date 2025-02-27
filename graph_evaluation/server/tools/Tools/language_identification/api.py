import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lang_2_0(accept: str='application/json', doc: str=None, txt: str='Last Friday we watched a terribly bad movie called "Kárate a muerte en Torremolinos", by an unknown Spanish director.', url: str=None, threshold: int=100, selection: str=None, verbose: str='n', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Language Identification 2.0"
    doc: Input file with the content to analyze. The supported formats for file contents can be found at https://www.meaningcloud.com/developer/documentation/supported-formats, (if 'doc' has a value, 'url' and 'txt' must be empty)
        txt: Input text. It can be plain text, HTML or XML, always using UTF-8 encoding, (if 'txt' has a value, 'url' and 'doc' must be empty)
        url: URL of the content to analyze. Currently only non-authenticated HTTP and FTP are supported. The content types supported for URL contents can be found at https://www.meaningcloud.com/developer/documentation/supported-formats, (if 'url' has a value, 'txt' and 'doc' must be empty)
        threshold: Language detection threshold as a percentage of similarity with respect to the top result
        selection: List of expected languages, separated by |.
        verbose: When active, it shows additional information about the languages detected.
        
    """
    url = f"https://language-identification.p.rapidapi.com/lang-2.0"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    if doc:
        querystring['doc'] = doc
    if txt:
        querystring['txt'] = txt
    if url:
        querystring['url'] = url
    if threshold:
        querystring['threshold'] = threshold
    if selection:
        querystring['selection'] = selection
    if verbose:
        querystring['verbose'] = verbose
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "language-identification.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

