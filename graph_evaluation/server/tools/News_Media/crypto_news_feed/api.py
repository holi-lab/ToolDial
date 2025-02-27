import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crypto_news_sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all news sources"
    
    """
    url = f"https://crypto-news-feed1.p.rapidapi.com/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-feed1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_individual_crypto_news_source_articles(newssourceid: str, query: str='bitcoin,ethereum,nft,cardano', limit: int=16, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return only the articles from a specific source"
    query: Only articles with titles including the search terms will be returned
        limit: limit the amount of returned articles
        
    """
    url = f"https://crypto-news-feed1.p.rapidapi.com/source/{newssourceid}"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-feed1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_news_feed(sources: str='nytimes,economictimes,newsbtc,coindesk,businessinsider,cointelegraph,coindesk', exclude: str='utoday,coinrivet', limit: int=100, query: str='bitcoin,ethereum,nft,cardano', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all the most recent articles from all sources"
    sources: Array of news sources to filter on.
Only articles from the selected sources will be returned.
        exclude: Array of news sources to exclude.
All articles but the ones from these sources will be returned
        limit: limit the amount of returned articles
        query: Only articles with titles including the search terms will be returned
        
    """
    url = f"https://crypto-news-feed1.p.rapidapi.com/feed"
    querystring = {}
    if sources:
        querystring['sources'] = sources
    if exclude:
        querystring['exclude'] = exclude
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-feed1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

