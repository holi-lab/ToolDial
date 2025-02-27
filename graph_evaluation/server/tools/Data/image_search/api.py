import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_search(q: str, cursor: str=None, hl: str='en', suggestion_query: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image Search"
    q: Search Query
        cursor: Cursor
        hl: Language Code
        suggestion_query: Suggestion Query
        
    """
    url = f"https://image-search13.p.rapidapi.com/"
    querystring = {'q': q, }
    if cursor:
        querystring['cursor'] = cursor
    if hl:
        querystring['hl'] = hl
    if suggestion_query:
        querystring['suggestion_query'] = suggestion_query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-search13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

