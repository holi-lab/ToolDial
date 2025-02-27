import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def data(quotesymbol: str, exchangeid: str=None, limit: str=None, basesymbol: str='BTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    quotesymbol: Quote Symbol. Example : USD, SGD, AUD
        basesymbol: Inset Symbol for search. Example : BTC, ETH
        
    """
    url = f"https://devandro-coin-market-v1.p.rapidapi.com/"
    querystring = {'quotesymbol': quotesymbol, }
    if exchangeid:
        querystring['exchangeId'] = exchangeid
    if limit:
        querystring['limit'] = limit
    if basesymbol:
        querystring['basesymbol'] = basesymbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "devandro-coin-market-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

