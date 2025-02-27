import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def risk_free_rate(duration: str='3m', geography: str='US', date: str='2023-05-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the risk free rate for a specific date. Default values are:
		
		date = yesterday
		geography = US
		duration = 3m
		
		Currently only US is supported. Durations supported are 3m, 5y and 10y. Historical values are supported until 1960-01-04."
    
    """
    url = f"https://risk-free-rate-api.p.rapidapi.com/api/risk_free_rate"
    querystring = {}
    if duration:
        querystring['duration'] = duration
    if geography:
        querystring['geography'] = geography
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "risk-free-rate-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

