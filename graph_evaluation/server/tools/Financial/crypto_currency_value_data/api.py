import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_coin_values(coin_code: str, limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Coin Values Api will return the current value for the particular requested Coin Code.
		
		Coin Code value can get from previous Fetch Coins Api.
		Example - 
		B-BTC_BUSD
		I-BTC_INR
		I-MANA_INR
		
		etc..."
    
    """
    url = f"https://crypto-currency-value-data.p.rapidapi.com/api/fetchcoinvalue"
    querystring = {'coin_code': coin_code, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-currency-value-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_coins(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Fetch Coins API will return a list of existing Coins with Pair Value."
    
    """
    url = f"https://crypto-currency-value-data.p.rapidapi.com/api/fetchccoin"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-currency-value-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

