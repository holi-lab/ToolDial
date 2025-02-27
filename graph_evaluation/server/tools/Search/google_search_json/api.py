import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_image(q: str, num: int=10, lr: str='lang_en', start: int=0, gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to perform image search using the Google Search JSON API. You can send a request with parameters like **q** (search query), **num** (desired number of images), **start** (starting index of results), and others. The request will generate a JSON response containing information about images that match the search query"
    q: Search query
        num: Number of search results to return.
Valid values are integers between 1 and 20, inclusive
        lr: Restricts the search to documents written in a particular language
        start: The index of the first result to return
        gl: The **gl** parameter value is a two-letter country code. The **gl** parameter boosts search results whose country of origin matches the parameter value.


        
    """
    url = f"https://google-search-json.p.rapidapi.com/search/image"
    querystring = {'q': q, }
    if num:
        querystring['num'] = num
    if lr:
        querystring['lr'] = lr
    if start:
        querystring['start'] = start
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-json.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_web(q: str, num: int=10, start: int=0, gl: str='US', lr: str='lang_en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to perform text search on the web using the Google Search JSON API. You can send a request with various parameters such as **q** (search query), **num** (desired number of results), **start** (starting index of results),  and more. The request will return search results in JSON format containing information such as titles, URLs, and descriptions of the search results."
    q: Search query
        num: Number of search results to return.
Valid values are integers between 1 and 20, inclusive
        start: The index of the first result to return
        gl: The **gl** parameter value is a two-letter country code. The **gl** parameter boosts search results whose country of origin matches the parameter value.


        lr: Restricts the search to documents written in a particular language
        
    """
    url = f"https://google-search-json.p.rapidapi.com/search/web"
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
            "X-RapidAPI-Host": "google-search-json.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

