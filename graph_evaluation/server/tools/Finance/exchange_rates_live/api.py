import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_individual_bank(bankid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rates from Central banks using prefixed. 
		Denmark National Bank (dkk), 
		National Bank Of Poland (pln),
		European Central Bank  (eur),
		European Central Bank  - calculated for USD base(usd), 
		Swedish Central Bank  (sek)
		Feel free to contact me If you wish to add new currency rates/ banks."
    
    """
    url = f"https://exchange-rates-live.p.rapidapi.com/rates/{bankid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rates-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_currency_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all rates from banks."
    
    """
    url = f"https://exchange-rates-live.p.rapidapi.com/rates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rates-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

