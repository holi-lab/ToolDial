import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_individual_news_source(newspaperid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news about crypto from a specific news source. There are 18 news sources to choose from cryptonews, coindesk, crypto.news, yahoo, fxstreet, cointelegraph, businessinsider, cryptopotato, ambcrypto, cryptonewsline, cryptoglobe, coingecko, cryptoslate, bitcoinist, forbes, nulltx, blockonomi, and coinspeaker."
    
    """
    url = f"https://crypto-news34.p.rapidapi.com/news/{newspaperid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def general_search(searchid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Can take a single word search that will return any articles with the given keyword."
    
    """
    url = f"https://crypto-news34.p.rapidapi.com/{searchid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_article_keyword_search(newspaperid: str, searchid: str='bitcoin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes a title and keyword that will then return any articles from a specific news source."
    
    """
    url = f"https://crypto-news34.p.rapidapi.com/news/{newspaperid}/{searchid}"
    querystring = {}
    if searchid:
        querystring['searchId'] = searchid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_crypto_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all news with crypto and bitcoin from many different websites across the internet."
    
    """
    url = f"https://crypto-news34.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

