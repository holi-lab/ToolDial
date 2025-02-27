import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_blockchain_latest_news(per_page: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date Blockchain news available daily."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/news/altcoin"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_altcoin_latest_news(page: int=1, per_page: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date Altcoin news available daily."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/news/altcoin"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_defi_latest_news(per_page: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date DeFi news available daily."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/news/defi"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ethereum_latest_news(page: int=1, per_page: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date Ethereum  news available daily."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/news/ethereum"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nft_latest_news(per_page: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date NFT news available daily."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/news/nft"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_details(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get News article details by providing the slug of the article."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/article/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bitcoin_latest_news(page: int=1, per_page: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date Bitcoin news available daily."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/news/bitcoin"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_up_to_date_crypto_news(per_page: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all the latest news about NFT, Bitcoin, Ethereum, DeFi, Altcoin and Blockchain."
    
    """
    url = f"https://crypto-news-live11.p.rapidapi.com/all"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news-live11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

