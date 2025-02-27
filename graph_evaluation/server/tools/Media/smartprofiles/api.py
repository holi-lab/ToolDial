import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def focus_api(target: str, targettype: str=None, format: str=None, jsonp: str=None, classification: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    target: Text to be profiled. Can also be an URL, in which case text on the page or document (PDF, .doc). in the page is profiled.
        targettype: Determines whether the 'target' is text or a link to a page or a document. Possible values are 'text' for text mode, 'url' for link mode, 'pdf' for PDF mode or 'auto' for detecting the type of 'target' automatically. Default 'auto'.
        format: Output format. Possible values include ‘xml’ (default), ‘json’, ‘jsonp’
        jsonp: JSONP callback. Optional parameter, requires ‘jsonp’ as output format
        classification: Select classification used. Possible values include ‘focus100k’, ‘iabtier1’, ‘iabtier2’, ‘iabtier2plus’. Default 'focus100k’
        lang: Language of the profiled text. Possible values are 'en' for English, 'fi' for Finnish, 'sv' for Swedish and 'de' for German. Default 'en'. Value 'auto’ attempts to detect the language automatically.
        
    """
    url = f"https://leiki-analyse-v1.p.rapidapi.com/focus/api"
    querystring = {'target': target, }
    if targettype:
        querystring['targettype'] = targettype
    if format:
        querystring['format'] = format
    if jsonp:
        querystring['jsonp'] = jsonp
    if classification:
        querystring['classification'] = classification
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "leiki-analyse-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

