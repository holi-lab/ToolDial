import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_providers(lang: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of news providers. Optional query string parameter "lang" (1-English, 2-Sinhala, 3- Tamil) can be sent to get providers belonging to a language"
    
    """
    url = f"https://sri-lankan-news-english-tamil-sinhala1.p.rapidapi.com/api/news/providers"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lankan-news-english-tamil-sinhala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(search: str=None, provider: int=1, pagecount: int=10, lang: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns 10 latest news belonging to 3 different languages, Optional query string parameters "pageCount", "provider","search", "lang" (1-English, 2-Sinhala, 3-Tamil) can be sent for more specific results"
    
    """
    url = f"https://sri-lankan-news-english-tamil-sinhala1.p.rapidapi.com/api/news"
    querystring = {}
    if search:
        querystring['search'] = search
    if provider:
        querystring['provider'] = provider
    if pagecount:
        querystring['pageCount'] = pagecount
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lankan-news-english-tamil-sinhala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

