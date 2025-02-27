import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def primary_endpoint(ticker: str, key: str, docs: str='10-k', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to gather 8 years (average) of financial data on the given ticker.  An average of 200 data points are provided for each year."
    
    """
    url = f"https://ase-data.p.rapidapi.com/api/stock?ticker=tsla&docs=10-k&key="
    querystring = {'ticker': ticker, 'key': key, }
    if docs:
        querystring['docs'] = docs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ase-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

