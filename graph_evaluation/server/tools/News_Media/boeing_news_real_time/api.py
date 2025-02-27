import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_boeing_news_articles(keywords: str, numberofarticles: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns filtered news articles. You just need to enter {keywords} after search/ and {numberOfArticles} after show/ for filtered results by keyword and showing certain number of news articles."
    
    """
    url = f"https://boeing-news-real-time.p.rapidapi.com/news/search/{keywords}/show/{numberofarticles}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boeing-news-real-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_boeing_news_articles(numberofarticles: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all Boeing news articles. Enter {numberOfArticles} after show/ to display a certain number of news articles."
    
    """
    url = f"https://boeing-news-real-time.p.rapidapi.com/news/show/{numberofarticles}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "boeing-news-real-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

