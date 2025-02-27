import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getimagesearchresults(term: str, region: str, safesearch: str, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of images from the search results with pagination."
    term: Keyword you want to search on DuckDuckGo for images.
        region: Region that you want to initiate the search.
        safesearch: Search safety parameter.
        offset: The number to offset the search results.
        
    """
    url = f"https://duckduckgo10.p.rapidapi.com/search/images"
    querystring = {'term': term, 'region': region, 'safeSearch': safesearch, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duckduckgo10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsearchresults(term: str, safesearch: str, region: str, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of URLs from the search results with pagination."
    term: Keyword you want to search on DuckDuckGo.
        safesearch: Search safety parameter.
        region: Region that you want to initiate the search.
        offset: The number to offset the search results.
        
    """
    url = f"https://duckduckgo10.p.rapidapi.com/search"
    querystring = {'term': term, 'safeSearch': safesearch, 'region': region, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duckduckgo10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvideosearchresults(safesearch: str, term: str, region: str, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of videos from the search results with pagination."
    safesearch: Search safety parameter.
        term: Keyword you want to search on DuckDuckGo for videos.
        region: Region that you want to initiate the search.
        offset: The number to offset the search results.
        
    """
    url = f"https://duckduckgo10.p.rapidapi.com/search/videos"
    querystring = {'safeSearch': safesearch, 'term': term, 'region': region, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duckduckgo10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnewssearchresults(term: str, region: str, safesearch: str, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of news from the search results with pagination."
    term: Keyword you want to search on DuckDuckGo for news.
        region: Region that you want to initiate the search.
        safesearch: Search safety parameter.
        offset: The number to offset the search results.
        
    """
    url = f"https://duckduckgo10.p.rapidapi.com/search/news"
    querystring = {'term': term, 'region': region, 'safeSearch': safesearch, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duckduckgo10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

