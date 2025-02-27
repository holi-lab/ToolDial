import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpositionsbytraderid(traderid: str, type: str='USDM', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the open positions by user's trader ID"
    traderid: Trader ID
        type: Type of grid
        
    """
    url = f"https://binance-futures-leaderboard3.p.rapidapi.com/user/{traderid}/positions"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuser(traderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all base information of User"
    traderid: Trader ID
        
    """
    url = f"https://binance-futures-leaderboard3.p.rapidapi.com/user/{traderid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserstatistics(traderid: str, type: str='USDM', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all statistics information of User by type"
    traderid: Trader ID
        type: Type of grid
        
    """
    url = f"https://binance-futures-leaderboard3.p.rapidapi.com/user/{traderid}/statistics"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getusers(sort: str='pnl', time: str='daily', type: str='USDM', trader: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns users that applies into the search query"
    sort: Sort by
        time: Time of the leaderboard search
        type: Type of grid
        trader: Trader Type
        
    """
    url = f"https://binance-futures-leaderboard3.p.rapidapi.com/users"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if time:
        querystring['time'] = time
    if type:
        querystring['type'] = type
    if trader:
        querystring['trader'] = trader
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettrendingmarkets(type: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns trending markets that applies into the search query"
    type: Type of markets
        page: Page of the list
        
    """
    url = f"https://binance-futures-leaderboard3.p.rapidapi.com/strategy/markets"
    querystring = {'type': type, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvolatility(type: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns most volatile pairs that applies into the search query"
    type: Type of markets
        page: Page of the list
        
    """
    url = f"https://binance-futures-leaderboard3.p.rapidapi.com/strategy/volatility"
    querystring = {'type': type, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstrategies(page: int=1, sort: str='roi', type: str='SPOT', market: str='ADA-BNB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns strategies that applies into the search query"
    page: Page of the list
        sort: Sorting values for the results
        type: Type of grid
        market: Market you want to search for strategies
        
    """
    url = f"https://binance-futures-leaderboard3.p.rapidapi.com/strategy"
    querystring = {}
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if type:
        querystring['type'] = type
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

