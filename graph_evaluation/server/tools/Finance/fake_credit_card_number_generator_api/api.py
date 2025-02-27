import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fake_credit_card_number_generator(cardnetwork: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass in one of the following card networks as a parameter:
		
		- amex
		- diners
		- discover
		- jcb
		- mastercard
		- visa"
    
    """
    url = f"https://fake-credit-card-number-generator-api.p.rapidapi.com/creditcard-cardgenerate/{cardnetwork}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-credit-card-number-generator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

