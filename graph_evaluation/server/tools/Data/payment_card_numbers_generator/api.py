import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate(quantity: int=5, scheme: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Generate payment card numbers
		
		## Default values
		
		**scheme**: "visa"
		**count**: 5"
    
    """
    url = f"https://payment-card-numbers-generator.p.rapidapi.com/generate"
    querystring = {}
    if quantity:
        querystring['quantity'] = quantity
    if scheme:
        querystring['scheme'] = scheme
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "payment-card-numbers-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

