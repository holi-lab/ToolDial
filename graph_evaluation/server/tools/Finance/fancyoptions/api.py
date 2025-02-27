import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def options_chain(ticker: str, expiration: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the an options chain for a given ticker and expiration date.
		
		Date format.  YYYY-MM-DD"
    ticker: A stock ticker
        expiration: expiration format: YYYY-MM-DD

ex: 2022-09-16
        
    """
    url = f"https://fancyoptions.p.rapidapi.com/options/{ticker}/expiration/{expiration}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fancyoptions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_expirations(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the expirations for a given ticker."
    
    """
    url = f"https://fancyoptions.p.rapidapi.com/options/expirations/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fancyoptions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_vertical(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns verticals for an individual ticker. The verticals that are returned have a 70% chance of being out of the money at time of expirations."
    
    """
    url = f"https://fancyoptions.p.rapidapi.com/verticals/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fancyoptions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s_p_500_verticals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the call and put spreads from the S&P 500 with the highest credits. The verticals that are returned have a 70% chance of being out of the money at time of expirations. This endpoint is updated hourly."
    
    """
    url = f"https://fancyoptions.p.rapidapi.com/verticals"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fancyoptions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

