import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stock_list_based_on_technical_indicators(indicators: str, date: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Returns all the LSE (London Stock Exchange) tickers that conform to the technical indicators that are applied.**
		
		More than 1 indicator can be applied"
    indicators: ### One or more technical indicators with , as delimiter

#### Example: 
- MACD_BEARISH_CROSSOVER, DOJI
- PRICE_CROSSED_ABOVE_UPPER_BB, DOJI
- TRIX_CROSSED_ABOVE_0
        date: Trade Date in YYYY-MM-DD format
**default:** picks last completed trade date
        format: Default Value: json
        
    """
    url = f"https://tradeindicators-io-lse-london-stock-exchange.p.rapidapi.com/query"
    querystring = {'indicators': indicators, }
    if date:
        querystring['date'] = date
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradeindicators-io-lse-london-stock-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

