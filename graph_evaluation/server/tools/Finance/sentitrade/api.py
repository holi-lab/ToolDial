import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def daily_sentiment(symbol: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns asset ticker queried, sentiment score, date queried, and number of articles used to calculate the sentiment score."
    
    """
    url = f"https://sentitrade.p.rapidapi.com/endpoint"
    querystring = {'symbol': symbol, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentitrade.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

