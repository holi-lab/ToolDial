import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def company(ssymbol: str='AAPL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Company details based on stock exchange symbols"
    
    """
    url = f"https://leibniz.p.rapidapi.com/company/"
    querystring = {}
    if ssymbol:
        querystring['ssymbol'] = ssymbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "leibniz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history(longma: str='30', pstart: str='20200723', chartint: str='5m', asset: str='MSFT', pend: str='20200724', shortma: str='9', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show data about stock exchange symbols, with moving averages, signals and backtesting"
    
    """
    url = f"https://leibniz.p.rapidapi.com/history/"
    querystring = {}
    if longma:
        querystring['longma'] = longma
    if pstart:
        querystring['pstart'] = pstart
    if chartint:
        querystring['chartint'] = chartint
    if asset:
        querystring['asset'] = asset
    if pend:
        querystring['pend'] = pend
    if shortma:
        querystring['shortma'] = shortma
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "leibniz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

