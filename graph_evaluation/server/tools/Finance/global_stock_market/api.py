import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stock_information(stock_ticker: str, stock_market: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one stock information from a specific market that could be:
		- BVMF (B3 Bovespa)
		- NYSE
		- Nasdaq"
    
    """
    url = f"https://global-stock-market.p.rapidapi.com/{stock_market}/{stock_ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

