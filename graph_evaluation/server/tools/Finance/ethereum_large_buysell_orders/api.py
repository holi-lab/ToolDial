import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getlargeorders(contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get large orders of Eth tokens"
    
    """
    url = f"https://ethereum-large-buy-sell-orders.p.rapidapi.com/large-orders/{contract_address}"
    querystring = {}
    if contract_address:
        querystring['contract_address'] = contract_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ethereum-large-buy-sell-orders.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

