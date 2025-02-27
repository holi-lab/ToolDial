import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def indianmarketnews(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest Indian Stock Market News from Yahoo Finance!. 
		Enter the stock Symbol without .NS. For example: Enter only HDFCBANK in the search string."
    
    """
    url = f"https://yahoo-finance-india1.p.rapidapi.com/market_india/news/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance-india1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

