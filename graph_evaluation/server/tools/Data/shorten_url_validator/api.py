import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shortenurlvalidator(url: str, email: str='myemail@validate.me', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to validate provided shorten URL. 
		Required parameter: URL
		Optional parameter: Email address"
    
    """
    url = f"https://shorten-url-validator.p.rapidapi.com/api/shortenurlvalidator"
    querystring = {'url': url, }
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shorten-url-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

