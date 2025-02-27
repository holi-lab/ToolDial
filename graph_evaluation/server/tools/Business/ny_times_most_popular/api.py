import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def viewed_period_json(period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of the most viewed articles on NYTimes.com for specified period of time (1 day, 7 days, or 30 days).
		"
    period: Time period: 1, 7, or 30 days.
        
    """
    url = f"https://ny-times-most-popular.p.rapidapi.com/viewed/{period}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-most-popular.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailed_period_json(period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of the most emailed articles on NYTimes.com for specified period of time (1 day, 7 days, or 30 days).
		"
    period: Time period: 1, 7, or 30 days.
        
    """
    url = f"https://ny-times-most-popular.p.rapidapi.com/emailed/{period}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-most-popular.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shared_period_json(period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of the most shared articles on NYTimes.com for specified period of time (1 day, 7 days, or 30 days).
		"
    period: Time period: 1, 7, or 30 days.
        
    """
    url = f"https://ny-times-most-popular.p.rapidapi.com/shared/{period}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-most-popular.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shared_period_share_type_json(share_type: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of the most shared articles by share type on NYTimes.com for specified period of time (1 day, 7 days, or 30 days).
		"
    share_type: Share type: facebook.
        period: Time period: 1, 7, or 30 days.
        
    """
    url = f"https://ny-times-most-popular.p.rapidapi.com/shared/{period}/{share_type}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-most-popular.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

