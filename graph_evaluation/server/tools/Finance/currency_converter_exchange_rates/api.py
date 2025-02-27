import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert(is_from: str='USD', amount: int=1, to: str='CHF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert any amount from a currency to another currency (or to multiple currencies)."
    is_from: A currency code to convert from. It's usually a 3-letter code except for DodgeCoin and DASH. Check the full list of supported currencies on the [About](https://rapidapi.com/apisbox/api/currency-converter-exchange-rates1/details) page.
        amount: The amount that you need to convert. (Default is 1.00)
        to: A currency code to convert to (separate codes with comma to convert to multiple currencies). Leave it empty to convert to all supported currencies.
        
    """
    url = f"https://currency-converter-exchange-rates1.p.rapidapi.com/convert"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if amount:
        querystring['amount'] = amount
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-exchange-rates1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

