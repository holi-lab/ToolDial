import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_text_from_number(language: str, number: int, currency: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the text base from the number.
		Languages: en, fr, de, ro, es, pt, hu, it, da, pl"
    
    """
    url = f"https://number2text1.p.rapidapi.com/getTextFromNumber"
    querystring = {'language': language, 'number': number, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "number2text1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

