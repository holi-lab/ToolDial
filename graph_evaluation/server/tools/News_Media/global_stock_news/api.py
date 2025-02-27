import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_news_with_translation(target_lang: str, nextpage: int=None, symbol: str='AAPL, AMZN, MSFT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search live news articles by stock symbols and also get translations of your specified target language. 10 articles are returned per request"
    target_lang: An ISO language code to translate the articles
        
    """
    url = f"https://global-stock-news.p.rapidapi.com/feed/translated"
    querystring = {'target_lang': target_lang, }
    if nextpage:
        querystring['nextPage'] = nextpage
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_news_english(nextpage: int=None, symbol: str='AAPL, AMZN, MSFT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search live news articles by stock symbols (English). 10 articles are returned per request."
    nextpage: Article ID to search from. Articles are returned in the order of the published time.
        symbol: Stock symbols to search, separated by comma
        
    """
    url = f"https://global-stock-news.p.rapidapi.com/feed"
    querystring = {}
    if nextpage:
        querystring['nextPage'] = nextpage
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

