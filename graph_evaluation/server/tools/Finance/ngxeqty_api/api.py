import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def eqtybyyear(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a Year from 2019- 2022 to get Market data with Symbol, Name, Close Price, Market Capitalization, Volume and Value e.g. Year=2022"
    
    """
    url = f"https://ngxeqty-api.p.rapidapi.com/EQTYByYear"
    querystring = {'Year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ngxeqty-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eqtybysymbol(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the Market Data Close Price, ISIN, Market Cap, Volume, Value etc. e.g. Symbol=MTNN' or Symbol='GTCO'"
    
    """
    url = f"https://ngxeqty-api.p.rapidapi.com/EQTYBySymbol"
    querystring = {'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ngxeqty-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

