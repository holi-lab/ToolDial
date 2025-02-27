import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lookup(ean: int, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product documents by UPC/EAN"
    ean: Add a valid EAN/UPC number
        language: Add your primary language "fr", "de" etc.. The fallback language is always set to "en"
        
    """
    url = f"https://singor-user-manual-v1.p.rapidapi.com/lookup"
    querystring = {'ean': ean, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "singor-user-manual-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

