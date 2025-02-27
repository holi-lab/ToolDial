import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_dollar_exchange_rates_from_major_mexican_banks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the Dollar Exchange Rates from some of the major Mexican and non-Mexican banks such as: BBVA, Banorte, Banregio, Banco Azteca, Banamex, Banxico, Santander, Scotiabank."
    
    """
    url = f"https://mexican-banks-dollar-exchange-rate.p.rapidapi.com/api/currency"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mexican-banks-dollar-exchange-rate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

