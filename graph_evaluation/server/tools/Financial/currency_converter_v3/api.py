import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def converter(target: str, source: str, amount: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to perform currency conversion between multiple currencies. The endpoint accepts input parameters in the form of the amount of money, source currency, and one or more target currencies."
    target: This parameter specifies the currency code of the currency that you want to convert to. It is a mandatory parameter and should be a valid three-letter currency code. If there are multiple target currencies, they should be separated by a comma (',') .
        source: This parameter represents the currency from which you want to convert. It is a mandatory parameter and should be a valid three-letter currency code, such as USD (United States Dollar) or EUR (Euro) or many others.
        amount: This parameter represents the amount that you want to convert from the source currency to the target currency. It is an optional parameter, and if not provided, the default value will be set to 1
        
    """
    url = f"https://currency-converter219.p.rapidapi.com/converter"
    querystring = {'target': target, 'source': source, }
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter219.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to retrieve all supported currencies."
    
    """
    url = f"https://currency-converter219.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter219.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

