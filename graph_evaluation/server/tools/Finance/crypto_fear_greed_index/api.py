import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index(limit: int=10, timestamp: str='1518048000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract Fear & Greed index value(s)
		
		Query params:
		limit (NUMBER) - Get latest N records (Optional)
		timestamp (STRING) - Get index value by UTC timestamp (Optional)
		
		If no query params specified, all available history will be fetched."
    limit: Limit number of last N records
        timestamp: Get fear & greed index value by specified utc timestamp
        
    """
    url = f"https://crypto-fear-greed-index2.p.rapidapi.com/index"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if timestamp:
        querystring['timestamp'] = timestamp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-fear-greed-index2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

