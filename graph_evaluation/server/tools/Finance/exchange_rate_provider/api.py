import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_foreign_exchange_rates_for_other_major_currencies(currencyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Foreign Exchange Rates for other major currencies. At the moment only GBP and USD are available. More will be added with future updates.
		
		So as `currencyId` please **use only** `/gbp` or `/usd`."
    
    """
    url = f"https://exchange-rate-provider.p.rapidapi.com/other/{currencyid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rate-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_eur_foreign_exchange_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Foreign Exchange Rates for default currency EUR."
    
    """
    url = f"https://exchange-rate-provider.p.rapidapi.com/eur"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rate-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

