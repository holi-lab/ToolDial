import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def climate_change_news_sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all news sources"
    
    """
    url = f"https://climate-change-news-feed1.p.rapidapi.com/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-change-news-feed1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_individual_crypto_news_source_articles(newssourceid: str, query: str='cop26,climate,carbon', limit: int=16, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return only the articles from a specific source"
    query: Only articles with titles including the search terms will be returned
        limit: Limit the amount of returned articles
        
    """
    url = f"https://climate-change-news-feed1.p.rapidapi.com/source/{newssourceid}"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-change-news-feed1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def climate_change_news_feed(sources: str='thetimes,guardian,nytimes,,dailymail,nypost,bbc,telegraph', exclude: str='thesun', query: str='cop26,climate,carbon', limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all the most recent articles from all sources"
    
    """
    url = f"https://climate-change-news-feed1.p.rapidapi.com/feed"
    querystring = {}
    if sources:
        querystring['sources'] = sources
    if exclude:
        querystring['exclude'] = exclude
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-change-news-feed1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

