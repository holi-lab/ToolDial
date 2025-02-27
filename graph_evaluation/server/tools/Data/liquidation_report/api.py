import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lickhunter_pro(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Suites of data tailored specifically for Bybit Lickhunter v4"
    
    """
    url = f"https://liquidation-report.p.rapidapi.com/lickhunterpro"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "liquidation-report.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def liquidation_data(coin: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the summary liquidation data for all coin. Data include total liquidation value, total liquidation amount, mean & median liquidation value."
    coin: Need to specify coin name if type equal to detail
        type: Choose type of data. If type equal to **detail**, it will show only 1 coin data. type need to be use with coin
        
    """
    url = f"https://liquidation-report.p.rapidapi.com/data"
    querystring = {}
    if coin:
        querystring['coin'] = coin
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "liquidation-report.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def liquidation_report(coin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest liquidation data from Binance, Bybit and OKX exchange. Data is limited to 1000 rows."
    coin: Not required parameter. If put will show only that coin related data. For example **BTC**
        
    """
    url = f"https://liquidation-report.p.rapidapi.com/report"
    querystring = {}
    if coin:
        querystring['coin'] = coin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "liquidation-report.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

