import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new_born_tokens_latest(page: str=None, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get new born latest  meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/newborn"
    querystring = {}
    if page:
        querystring['page'] = page
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_born_tokens_signalled(page: str=None, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get new born signalled  meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/newborn-signalled"
    querystring = {}
    if page:
        querystring['page'] = page
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_born_tokens_trending(page: str=None, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get new born trending  meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/newborn-trending"
    querystring = {}
    if page:
        querystring['page'] = page
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_time_top(network: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all time top meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/alltime"
    querystring = {}
    if network:
        querystring['network'] = network
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(network: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/trending"
    querystring = {}
    if network:
        querystring['network'] = network
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_top(network: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today top meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/daily-top"
    querystring = {}
    if network:
        querystring['network'] = network
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_24h_gainers(network: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 24h gainers meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/gainer"
    querystring = {}
    if network:
        querystring['network'] = network
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_traded(page: str='1', network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  most traded meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/most-traded"
    querystring = {}
    if page:
        querystring['page'] = page
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_listings(page: str='1', network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get freshly listed meme coins
		Possible filtering by network:
		binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain"
    network: Possible filtering by network:
binance, ethereum, arbitrum, polygon, avax, fantom, cronos, dogechain
        
    """
    url = f"https://meme-coins-crypto.p.rapidapi.com/new"
    querystring = {}
    if page:
        querystring['page'] = page
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-coins-crypto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

