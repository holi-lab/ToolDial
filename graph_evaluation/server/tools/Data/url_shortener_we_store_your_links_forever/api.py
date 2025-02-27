import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_request(shorturl: str, secretkey: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter your password to get click-through statistics for your link.
		Shows long URL and visitor statistics."
    
    """
    url = f"https://url-shortener-we-store-your-links-forever.p.rapidapi.com/"
    querystring = {'shortUrl': shorturl, }
    if secretkey:
        querystring['secretKey'] = secretkey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-shortener-we-store-your-links-forever.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

