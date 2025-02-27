import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_financial_time_series_daily(symbol: str, asset_class: str, size: str='full', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns synthetic financial time series (Date, Price Return) for the specified symbol. 
		
		Parameters:
		asset_class = [equity, bond, commodity, mixed]
		symbol = [0000...0999] 
		size = [compact, full]  (size = compact returns the most recent 252 data points)"
    
    """
    url = f"https://synthetic-financial-data.p.rapidapi.com/"
    querystring = {'symbol': symbol, 'asset_class': asset_class, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "synthetic-financial-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

