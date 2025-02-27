import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crypto_news(keyword: str='bitcoin', newspaper: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "aggregates crypto news from top sources, ensuring that you never miss a beat."
    
    """
    url = f"https://news-aggregator1.p.rapidapi.com/news/crypto/{keyword}/{newspaper}"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if newspaper:
        querystring['newspaper'] = newspaper
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-aggregator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

