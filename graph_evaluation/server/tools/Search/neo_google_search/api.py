import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def web_search(q: str, lr: str='lang_en', gl: str='US', num: int=10, start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used for web searches"
    q: Search query
        lr: Restricts the search to documents written in a particular language
        gl: The gl parameter value is a two-letter country code. The gl parameter boosts search results whose country of origin matches the parameter value
        num: Number of search results to return.
Valid values are integers between 1 and 20, inclusive
        start: The index of the first result to return.
        
    """
    url = f"https://neo-google-search.p.rapidapi.com/search"
    querystring = {'q': q, }
    if lr:
        querystring['lr'] = lr
    if gl:
        querystring['gl'] = gl
    if num:
        querystring['num'] = num
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "neo-google-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_search(q: str, gl: str='US', lr: str='lang_en', start: int=0, num: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used for image searches"
    q: Search query
        gl: The gl parameter value is a two-letter country code. The gl parameter boosts search results whose country of origin matches the parameter value
        lr: Restricts the search to documents written in a particular language
        start: The index of the first result to return.
        num: Number of search results to return.
Valid values are integers between 1 and 20, inclusive
        
    """
    url = f"https://neo-google-search.p.rapidapi.com/imagesearch"
    querystring = {'q': q, }
    if gl:
        querystring['gl'] = gl
    if lr:
        querystring['lr'] = lr
    if start:
        querystring['start'] = start
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "neo-google-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

