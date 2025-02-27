import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_news_from_one_news_source(page: int, source: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news from one news source passing {source} paramter value and {page} value.Each news soure different number of news count in one page.
		
		
		**News Sources
		--Lankadeepa.lk
		--Deshaya.lk
		--Ada.lk
		--BBCSinhala.com
		--Mawbima.lk"
    
    """
    url = f"https://sri-lanka-news-api.p.rapidapi.com/news/sinhala/{source}/data/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_latest_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest news from all the news sources.This route return all the latest news from each news source at once"
    
    """
    url = f"https://sri-lanka-news-api.p.rapidapi.com/news/sinhala"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sri-lanka-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

