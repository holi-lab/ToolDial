import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_nft_and_decentralized_finance_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all news about NFT and Decentralized Finance (DeFi) from various news resources like Cryptonews, CoinTelegraph, NewsBTC from all over the world."
    
    """
    url = f"https://cryptocurrency-news-live1.p.rapidapi.com/NFT_DeFi"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-news-live1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cryptocurrency_news_from_individual_source(newspapername: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Cryptocurrency News articles from a specific news resource"
    
    """
    url = f"https://cryptocurrency-news-live1.p.rapidapi.com/news/{newspapername}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-news-live1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_cryptocurrency_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back  all news articles about Cryptocurrency from various news sources like CoinDesk, CoinTelegraph, Cryptonews, NewsBTC, TheEconomicTimes from all over the world"
    
    """
    url = f"https://cryptocurrency-news-live1.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-news-live1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

