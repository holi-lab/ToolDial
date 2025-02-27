import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listquotes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the available quotes in JSON Array this API support, all the available quotes can be used in source and destination quote. Refer exchange endpoint for more information how to call the currency exchange from the source quote to destination quote."
    
    """
    url = f"https://currency-exchange.p.rapidapi.com/listquotes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange(is_from: str, to: str, q: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Currency Exchange by specifying the quotes of source (from) and destination (to), and optionally the source amount to calculate which to get the destination amount, by default the source amount will be 1."
    from: Source Quote
        to: Destination Quote
        q: Source Amount
        
    """
    url = f"https://currency-exchange.p.rapidapi.com/exchange"
    querystring = {'from': is_from, 'to': to, }
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

