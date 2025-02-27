import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_currency_for_today(to: str, amount: int, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to convert currencies from one to another."
    
    """
    url = f"https://currency-converter73.p.rapidapi.com/"
    querystring = {'to': to, 'amount': amount, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter73.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_currency_with_date(amount: int, to: str, is_from: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to convert currencies from one to another."
    
    """
    url = f"https://currency-converter73.p.rapidapi.com/"
    querystring = {'amount': amount, 'to': to, 'from': is_from, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter73.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

