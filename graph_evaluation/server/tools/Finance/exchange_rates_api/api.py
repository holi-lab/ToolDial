import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets a list of available currency symbols along with their full names.
		
		GET /currencies HTTP/1.1"
    
    """
    url = f"https://exchange-rates-api2.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rates-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical(yyyy_mm_dd: str, to: str='EUR,USD', is_from: str='CHF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns historical rates for any working day since 4 January 1999.
		
		GET /1999-01-04 HTTP/1.1
		You can again tweak the response using the from and to parameters."
    
    """
    url = f"https://exchange-rates-api2.p.rapidapi.com/{yyyy_mm_dd}"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rates-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest(to: str='USD,AUD', is_from: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the latest rates.
		
		GET /latest HTTP/1.1
		Rates quote against the Euro by default. You can quote against other currencies using the from parameter.
		
		GET /latest?from=USD HTTP/1.1
		to limits returned rates to specified values.
		
		GET /latest?to=USD,GBP HTTP/1.1"
    
    """
    url = f"https://exchange-rates-api2.p.rapidapi.com/latest"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rates-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

