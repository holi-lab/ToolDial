import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1getfundhistoricalprices(isin: str, to: str='2020-12-31', is_from: str='2015-01-25', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fund's historical prices"
    isin: Fund's ISIN
        to: Finishing date. Format YYYY-MM-DD.
        is_from: Starting date. Format YYYY-MM-DD.
        
    """
    url = f"https://funds.p.rapidapi.com/v1/historicalPrices/{isin}"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1getfundlatestprice(isin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fund latest price"
    isin: Fund's ISIN
        
    """
    url = f"https://funds.p.rapidapi.com/v1/fund/{isin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

