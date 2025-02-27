import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_media_screening(name: str, webhook: str=None, searchtype: str=None, realtime: str=None, format: str=None, keyword: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint searches the person of interest in millions of external global news sources and returns the news articles in which person of interest was found with details, negative or positive sentiment and key connections found related to person of interest."
    
    """
    url = f"https://news-media-screening.p.rapidapi.com/ai/news"
    querystring = {'name': name, }
    if webhook:
        querystring['webhook'] = webhook
    if searchtype:
        querystring['searchtype'] = searchtype
    if realtime:
        querystring['realtime'] = realtime
    if format:
        querystring['format'] = format
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-media-screening.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

