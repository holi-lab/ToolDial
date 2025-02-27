import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def leveraged_equity_calculator(leverage: int, originalshareprice: int, originalequity: int, projectedshareprice: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a leveraged value, change in an underlying asset price and a starting equity amount; calculate the projected equity amount."
    
    """
    url = f"https://underlying-vs-theoretical-leveraged-etf-returns.p.rapidapi.com/api/etfwhatif"
    querystring = {'leverage': leverage, 'originalSharePrice': originalshareprice, 'originalEquity': originalequity, 'projectedSharePrice': projectedshareprice, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "underlying-vs-theoretical-leveraged-etf-returns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_etf(leverage: int, underlying: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a leverage value and an underlying asset percent change; returns the equity asset percent change."
    
    """
    url = f"https://underlying-vs-theoretical-leveraged-etf-returns.p.rapidapi.com/api/etf"
    querystring = {'leverage': leverage, 'underlying': underlying, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "underlying-vs-theoretical-leveraged-etf-returns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

