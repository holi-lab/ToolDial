import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def media_sources_statistics(apikey: str='b7ed776c-7f0f-4dd2-ba9d-9c6a6e1f44cb', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search and find details or statistics on media sources"
    
    """
    url = f"https://public-url-share.p.rapidapi.com/sources"
    querystring = {}
    if apikey:
        querystring['apiKey'] = apikey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "public-url-share.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_50_startups_news_from_the_last_month(is_from: str='2022-12-19', sourcegroup: str='top100', apikey: str='b7ed776c-7f0f-4dd2-ba9d-9c6a6e1f44cb', language: str='en', topic: str='Startups', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top 50 startups news from the last month"
    
    """
    url = f"https://public-url-share.p.rapidapi.com/all/"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if sourcegroup:
        querystring['sourceGroup'] = sourcegroup
    if apikey:
        querystring['apiKey'] = apikey
    if language:
        querystring['language'] = language
    if topic:
        querystring['topic'] = topic
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "public-url-share.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

