import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_prices(key: str=None, names: str=None, mids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get general price information per card"
    key: Your Cardsearch api key
        names: Single card: "Terror", card from set: "Terror@6ED"
        mids: A single card or multiple cards by multiverseid
        
    """
    url = f"https://jaap-cardsearch-v1.p.rapidapi.com/v1/prices"
    querystring = {}
    if key:
        querystring['key'] = key
    if names:
        querystring['names[]'] = names
    if mids:
        querystring['mids[]'] = mids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jaap-cardsearch-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

