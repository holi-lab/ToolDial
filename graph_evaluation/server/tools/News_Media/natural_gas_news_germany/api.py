import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_article_from_specific_news_source(source: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to query articles from a specific news source.
		
		Possible news sources include:
		
		- tagesschau
		- n-tv
		- faz
		- sueddeutsche
		- welt"
    
    """
    url = f"https://natural-gas-news-germany.p.rapidapi.com/api/v1/articles/{source}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "natural-gas-news-germany.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_articles_from_all_news_sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint, you will get the articles **from all the registered news sources** at once. Registered news sources are: Tagesschau, n-tv, Frankfurter Allgemeine Zeitung, Sueddeutsche and Welt."
    
    """
    url = f"https://natural-gas-news-germany.p.rapidapi.com/api/v1/articles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "natural-gas-news-germany.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

