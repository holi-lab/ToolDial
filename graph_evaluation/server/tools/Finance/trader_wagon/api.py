import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listportfolios(limit: int=10, portfoliotype: str=None, available: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List portfolios"
    limit: The number of results to return
        portfoliotype: Portfolio type
        available: List available portfolios
        
    """
    url = f"https://trader-wagon.p.rapidapi.com/listPortfolios"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if portfoliotype:
        querystring['portfolioType'] = portfoliotype
    if available:
        querystring['available'] = available
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trader-wagon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getportfolioinfo(portfolioid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get portfolio info of a given portfolio id"
    portfolioid: Portfolio ID of the trader
        
    """
    url = f"https://trader-wagon.p.rapidapi.com/getPortfolioInfo"
    querystring = {'portfolioId': portfolioid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trader-wagon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettraderpositions(portfolioid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trader positions of a given portfolio id"
    portfolioid: Portfolio ID of the trader
        
    """
    url = f"https://trader-wagon.p.rapidapi.com/getTraderPositions"
    querystring = {'portfolioId': portfolioid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trader-wagon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

