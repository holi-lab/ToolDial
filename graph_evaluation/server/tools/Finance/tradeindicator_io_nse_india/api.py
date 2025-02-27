import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stock_list_based_on_technical_indicators(indicators: str, format: str=None, options_only: bool=None, date: str='2022-12-26', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Returns all the NSE tickers that conform to the technical indicators that are applied.**
		
		More than 1 indicator can be applied"
    indicators: ### One or more technical indicators with , as delimiter

#### Example: 
- MACD_BEARISH_CROSSOVER, DOJI
- PRICE_CROSSED_ABOVE_UPPER_BB, DOJI
- TRIX_CROSSED_ABOVE_0
        date: Trade Date in YYYY-MM-DD format
**default:** picks last completed trade date
        
    """
    url = f"https://tradeindicator-io-nse-india.p.rapidapi.com/query"
    querystring = {'indicators': indicators, }
    if format:
        querystring['format'] = format
    if options_only:
        querystring['options_only'] = options_only
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradeindicator-io-nse-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

