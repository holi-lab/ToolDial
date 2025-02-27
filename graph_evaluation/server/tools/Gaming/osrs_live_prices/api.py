import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historical_prices(timestep: str, itemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives a list of the high and low prices of item with the given id at the given interval"
    
    """
    url = f"https://osrs-live-prices.p.rapidapi.com/api/v1/prices"
    querystring = {'timestep': timestep, 'itemID': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osrs-live-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_price(timestep: str, itemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gives the latest price for the given itemID and timestep"
    
    """
    url = f"https://osrs-live-prices.p.rapidapi.com/api/v1/latestprice"
    querystring = {'timestep': timestep, 'itemID': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osrs-live-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

