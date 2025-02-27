import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nft_news(nftnews: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is where you get all the up to date NFT news. This is a live feed and is updated daily."
    nftnews: This is where you get all the up to date NFT News. This is updated daily.
        
    """
    url = f"https://nft-api-news.p.rapidapi.com/nft"
    querystring = {}
    if nftnews:
        querystring['nftnews'] = nftnews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft-api-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def welcome(nft: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the welcome page."
    nft: This is the welcome page.
        
    """
    url = f"https://nft-api-news.p.rapidapi.com/nft"
    querystring = {}
    if nft:
        querystring['nft'] = nft
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft-api-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

