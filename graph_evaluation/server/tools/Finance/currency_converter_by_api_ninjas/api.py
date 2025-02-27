import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_convertcurrency(amount: int, have: str, want: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Convert Currency API endpoint."
    amount: Amount of currency to convert.
        have: Currency you currently hold. Must be 3-character currency code (e.g. **USD**).
        want: Currency you want to convert to. Must be 3-character currency code (e.g. **EUR**)
        
    """
    url = f"https://currency-converter-by-api-ninjas.p.rapidapi.com/v1/convertcurrency"
    querystring = {'amount': amount, 'have': have, 'want': want, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

