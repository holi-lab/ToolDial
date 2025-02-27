import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_languages(accept_encoding: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of supported languages for translation in **ISO-639-1**, **ISO-639-2** or **BCP-47** formats."
    
    """
    url = f"https://lecto-translation.p.rapidapi.com/v1/translate/languages"
    querystring = {}
    if accept_encoding:
        querystring['Accept-Encoding'] = accept_encoding
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lecto-translation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

