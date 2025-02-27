import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new(request: str, lang: str, country: str, cost: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "new"
    request: Just write the name of the product you want to sell, as your customer usually searches for it.
        lang: so far only english is available
        country: US/MX/CA/UK/AE/AU/DE/FR/ES/IT/IN/BR/NL/SE/PL/SG/JP
        cost:  Specify your SKU cost, including delivery, taxes, and customs payments
        
    """
    url = f"https://sales-forecast-for-amazon-sellers.p.rapidapi.com/"
    querystring = {'request': request, 'lang': lang, 'country': country, }
    if cost:
        querystring['cost'] = cost
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sales-forecast-for-amazon-sellers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

