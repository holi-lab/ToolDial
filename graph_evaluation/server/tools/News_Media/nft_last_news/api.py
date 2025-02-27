import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_individual_news_from_source(newssource: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets news about NFT from a specific news resource.
		
		/news/theblockcrypto
		
		/news/theverge
		
		/news/cryptonewsz
		
		/news/techstory
		
		/news/cointelegraph
		
		/news/coinpedia
		
		/news/dailymail
		
		/news/cryptonews
		
		/news/ndtv
		
		/news/marketwatch"
    
    """
    url = f"https://nft-last-news.p.rapidapi.com/news/{newssource}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft-last-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_last_nft_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back last NFT news from most famous finance websites and most famous NFT websites."
    
    """
    url = f"https://nft-last-news.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft-last-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

