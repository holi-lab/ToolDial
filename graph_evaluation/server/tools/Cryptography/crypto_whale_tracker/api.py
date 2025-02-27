import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address_history(chain: str, user_addr: str, page_count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get activity history of specific address
		
		Supported chains:
		eth, bsc, arb, op, matic, avax, ftm, xdai"
    
    """
    url = f"https://crypto-whale-tracker.p.rapidapi.com/history/list"
    querystring = {'chain': chain, 'user_addr': user_addr, }
    if page_count:
        querystring['page_count'] = page_count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_projects(chain: str, user_addr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get projects for specific address
		
		Supported chains:
		eth, bsc, arb, op, matic, avax, ftm, xdai"
    
    """
    url = f"https://crypto-whale-tracker.p.rapidapi.com/portfolio/project_list"
    querystring = {'chain': chain, 'user_addr': user_addr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_balance(user_addr: str, chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get balance for specific address
		
		Supported chains:
		eth, bsc, arb, op, matic, avax, ftm, xdai"
    
    """
    url = f"https://crypto-whale-tracker.p.rapidapi.com/token/balance_list"
    querystring = {'user_addr': user_addr, 'chain': chain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_net_worth_sparklines(user_addr: str='0x2a82ae142b2e62cb7d10b55e323acb1cab663a26', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sparklines 24h data for specific address to render chart"
    
    """
    url = f"https://crypto-whale-tracker.p.rapidapi.com/asset/net_curve_24h"
    querystring = {}
    if user_addr:
        querystring['user_addr'] = user_addr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_info(addr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info about specific address"
    
    """
    url = f"https://crypto-whale-tracker.p.rapidapi.com/user/addr"
    querystring = {'addr': addr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def whale_portfolios(order_by: str, start: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get huge whale portfolios
		pagination example: ?start=0&limit=20"
    
    """
    url = f"https://crypto-whale-tracker.p.rapidapi.com/whale/list"
    querystring = {'order_by': order_by, 'start': start, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trade_signals(start: str, limit: str, is_whale: str, order_by: str='time_at', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameters:
		sort_by: usd_value, time_at"
    
    """
    url = f"https://crypto-whale-tracker.p.rapidapi.com/activity/list"
    querystring = {'start': start, 'limit': limit, 'is_whale': is_whale, }
    if order_by:
        querystring['order_by'] = order_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

