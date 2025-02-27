import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_search(q: str, start: str='0', num: int=10, gl: str='us', lr: str='lang_en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used for image searches"
    start: The index of the first result to return.
        num: Number of search results to return.
Valid values are integers between 1 and 20, inclusive
        gl: The gl parameter value is a two-letter country code. The gl parameter boosts search results whose country of origin matches the parameter value.
        lr: Restricts the search to documents written in a particular language
        
    """
    url = f"https://google-search72.p.rapidapi.com/imagesearch"
    querystring = {'q': q, }
    if start:
        querystring['start'] = start
    if num:
        querystring['num'] = num
    if gl:
        querystring['gl'] = gl
    if lr:
        querystring['lr'] = lr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search72.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_search(q: str, num: int=10, start: str='0', gl: str='us', lr: str='lang_en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used for web searches"
    q: Search query
        num: Number of search results to return.
Valid values are integers between 1 and 20, inclusive
        start: The index of the first result to return.
        gl: The gl parameter value is a two-letter country code. The gl parameter boosts search results whose country of origin matches the parameter value.
        lr: Restricts the search to documents written in a particular language
        
    """
    url = f"https://google-search72.p.rapidapi.com/search"
    querystring = {'q': q, }
    if num:
        querystring['num'] = num
    if start:
        querystring['start'] = start
    if gl:
        querystring['gl'] = gl
    if lr:
        querystring['lr'] = lr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search72.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

