import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keyword_suggestion_get(phrase: str, lang: str='en', loc: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recommend keywords for SEO and SEM scored by relevance."
    lang: Languages available for targeting. Use the corresponding criterion ID for the Language target . 
        loc: Two-letter country codes. 
        
    """
    url = f"https://twinword-keyword-suggestion-v1.p.rapidapi.com/suggest/"
    querystring = {'phrase': phrase, }
    if lang:
        querystring['lang'] = lang
    if loc:
        querystring['loc'] = loc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-keyword-suggestion-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

