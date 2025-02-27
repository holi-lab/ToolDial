import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_auctions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Of Rarible marketplace, you will found open Auctions display most NFT listed, current endpoint offer your this Data to fetch"
    
    """
    url = f"https://rarible-data-scraper.p.rapidapi.com/auctions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rarible-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hot_bids(size: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Of Rarible marketplace, you will found Hot Bids section display most NFT listed, current endpoint offer your this Data to fetch"
    
    """
    url = f"https://rarible-data-scraper.p.rapidapi.com/hot_bids/{size}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rarible-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_top_sellers(day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current Endpoint offer you Data of best sellers Rarible platform by day set us params"
    
    """
    url = f"https://rarible-data-scraper.p.rapidapi.com/top_sellers/{day}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rarible-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_collection(day: int, size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**You can get the Top collection of Nfts by day and size selected:**
		`for example you can get best or Top collection of curent day, or of last 7 day`
		
		**Notice**: 'The maximum of size its 100'"
    
    """
    url = f"https://rarible-data-scraper.p.rapidapi.com/top_collection/{day}/{size}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rarible-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hot_collection(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Of rarible marketplace, you will found Hot Collection section display most NFT listed, current endpoint offer your this Data to fetch"
    
    """
    url = f"https://rarible-data-scraper.p.rapidapi.com/hot_collection"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rarible-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_top_buyers(day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current Endpoint offer you Data of best buyers Rarible platform by day set us params"
    
    """
    url = f"https://rarible-data-scraper.p.rapidapi.com/top_buyers/{day}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rarible-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

