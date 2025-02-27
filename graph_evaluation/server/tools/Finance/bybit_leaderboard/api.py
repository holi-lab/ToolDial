import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getservertime(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns server time."
    
    """
    url = f"https://bybit-leaderboard.p.rapidapi.com/GetServerTime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit-leaderboard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trader_details(user_id: str, product: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns full trader info."
    
    """
    url = f"https://bybit-leaderboard.p.rapidapi.com/trader/{user_id}/details"
    querystring = {'product': product, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit-leaderboard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trader_positions(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all trader positions."
    
    """
    url = f"https://bybit-leaderboard.p.rapidapi.com/trader/{user_id}/positions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit-leaderboard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traders_with_positions(period: str, metric: str, product: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns traders with shared positions only."
    
    """
    url = f"https://bybit-leaderboard.p.rapidapi.com/traders/with-positions"
    querystring = {'period': period, 'metric': metric, 'product': product, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit-leaderboard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_master_traders(page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns MASTER traders that match your search filters."
    
    """
    url = f"https://bybit-leaderboard.p.rapidapi.com/search/master_traders"
    querystring = {'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit-leaderboard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leaderboard(metric: str, product: str, page: int, period: str, display_shared_positions: bool, user_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns traders that match your search filters."
    display_shared_positions: search traders with shared positions
        
    """
    url = f"https://bybit-leaderboard.p.rapidapi.com/search/leaderboard"
    querystring = {'metric': metric, 'product': product, 'page': page, 'period': period, 'display_shared_positions': display_shared_positions, }
    if user_name:
        querystring['user_name'] = user_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bybit-leaderboard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

