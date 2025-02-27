import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_by_email(email: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get detail user informations, requires users's own cubiculus credential"
    email: users's email
        apikey: Application key. Accessing application is identified by key. This application key could be for free obtained at cubiculus.com after registration
        
    """
    url = f"https://cubiculuscollection.p.rapidapi.com/user-by-email/{apiKey}/{email}"
    querystring = {'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cubiculuscollection.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

