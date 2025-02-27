import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_arbitrage_opportunities(currencypair: str, minprofit: str='10', exchanges: str='coinbase,gateio', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It gets the arbitrage opportunities when provided with a mandatory currency pair.
		
		The opportunities can be filtered to certain exchanges by passing in the exchanges. The exchanges can be passed as 1 exchange value such as 'coinbase' or as comma separated values like 'coinbase,gateio' to represent a few exchanges
		The opportunities can be filtered to have a minimum profit by passing in the percentage profit.  The minimum profit is passed as '10' if a 10% profit is required"
    
    """
    url = f"https://crypto-arbitrage-opportunities.p.rapidapi.com/arbitrage-opportunities"
    querystring = {'currencyPair': currencypair, }
    if minprofit:
        querystring['minProfit'] = minprofit
    if exchanges:
        querystring['exchanges'] = exchanges
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-arbitrage-opportunities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

