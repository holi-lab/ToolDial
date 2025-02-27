import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def author(author: str, q: str, count: str='20', setlang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search news articles from a specific publisher"
    author: URL of a website
        q: Search query terms
        count: Number of results
        setlang: Language code
        
    """
    url = f"https://news-search9.p.rapidapi.com/author"
    querystring = {'author': author, 'q': q, }
    if count:
        querystring['count'] = count
    if setlang:
        querystring['setLang'] = setlang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-search9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, setlang: str='en', count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search news articles"
    q: Search query terms.  Optionally, add *site:example.com* to get articles from a specific publisher.
        setlang: Language code. Example: \"en\"
        count: Number of news articles to return
        
    """
    url = f"https://news-search9.p.rapidapi.com/news/search"
    querystring = {'q': q, }
    if setlang:
        querystring['setLang'] = setlang
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-search9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

