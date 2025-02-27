import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_currency_with_amount(fromcurrency: str, amount: str, tocurrency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert currency to another currency with a specific amount."
    
    """
    url = f"https://currency-converter25.p.rapidapi.com/limitless-plains-90877.herokuapp.com/rates/{fromcurrency}/to/{tocurrency}/{amount}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter25.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_currency(tocurrency: str, fromcurrency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert currency to another currency."
    
    """
    url = f"https://currency-converter25.p.rapidapi.com/limitless-plains-90877.herokuapp.com/rates/{fromcurrency}/to/{tocurrency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter25.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

