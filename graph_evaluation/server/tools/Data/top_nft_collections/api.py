import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending_with_pagination(limit: int=50, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trending with pagination provided by OpenSea"
    
    """
    url = f"https://top-nft-collections.p.rapidapi.com/api/nft-trending-rankings-opensea-pagination"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-nft-collections.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_testnets(sortby: str='ONE_HOUR_VOLUME', count: int=100, time: str='ONE_HOUR', chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Giving the same result in https://opensea.io/rankings/trending"
    
    """
    url = f"https://top-nft-collections.p.rapidapi.com/api/nft-trending-rankings-opensea-testnets"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if count:
        querystring['count'] = count
    if time:
        querystring['time'] = time
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-nft-collections.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(time: str='ONE_HOUR', chain: str=None, sortby: str='ONE_HOUR_VOLUME', count: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Giving the same result in https://opensea.io/rankings/trending"
    
    """
    url = f"https://top-nft-collections.p.rapidapi.com/api/nft-trending-rankings-opensea"
    querystring = {}
    if time:
        querystring['time'] = time
    if chain:
        querystring['chain'] = chain
    if sortby:
        querystring['sortBy'] = sortby
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-nft-collections.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_ranking_testnets(time: str='ONE_HOUR', count: int=100, sortby: str='ONE_HOUR_VOLUME', chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Giving the same result in https://opensea.io/rankings"
    
    """
    url = f"https://top-nft-collections.p.rapidapi.com/api/nft-top-rankings-opensea-testnets"
    querystring = {}
    if time:
        querystring['time'] = time
    if count:
        querystring['count'] = count
    if sortby:
        querystring['sortBy'] = sortby
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-nft-collections.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_ranking(sortby: str='ONE_HOUR_VOLUME', time: str='ONE_HOUR', chain: str=None, count: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Giving the same result in https://opensea.io/rankings"
    
    """
    url = f"https://top-nft-collections.p.rapidapi.com/api/nft-top-rankings-opensea"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if time:
        querystring['time'] = time
    if chain:
        querystring['chain'] = chain
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-nft-collections.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_nft_collections(order: str='desc', period: str='24h', sortby: str='volume', offset: int=0, network: str='Ethereum', limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the NFT collections ordered by volume or sales."
    order: Can be `desc` or `asc`
        period: Can be `24h` for 24 hours, `7d` for 7 days, `30d` for 30 days, or `all` for all time. Default is `all`
        sortby: How collections are sorted. Can be `volume`, `sales`, `floorPrice`, `averagePrice`, or `marketCap`. Default is `volume`
        offset: For pagination. Number of collections offset from the beginning of the result list.
        network: Can be `Ethereum`, `BNB Chain`, `Polygon`, `Arbitrum`, `Optimism`, `Solana`, or `null` for all top collections comparing all the above networks.
        limit: For pagination. Maximum number of contracts to return. Default is `20`, max is `100`.
        
    """
    url = f"https://top-nft-collections.p.rapidapi.com/api/get-collections-ranking"
    querystring = {}
    if order:
        querystring['order'] = order
    if period:
        querystring['period'] = period
    if sortby:
        querystring['sortBy'] = sortby
    if offset:
        querystring['offset'] = offset
    if network:
        querystring['network'] = network
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-nft-collections.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

