import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def devices(site: str, lang: str, currency: str, operator: str='unlocked', variation: str='8gb', model: str='iphone-3g', brand: str='apple', type: str='phone', condition: str='medium', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    lang: sv, dk, fi
        currency: SEK, EUR, DKK
        
    """
    url = f"https://pantaluren.p.rapidapi.com/api/devices"
    querystring = {'site': site, 'lang': lang, 'currency': currency, }
    if operator:
        querystring['operator'] = operator
    if variation:
        querystring['variation'] = variation
    if model:
        querystring['model'] = model
    if brand:
        querystring['brand'] = brand
    if type:
        querystring['type'] = type
    if condition:
        querystring['condition'] = condition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pantaluren.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

