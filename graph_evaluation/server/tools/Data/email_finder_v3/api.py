import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_finder(firstname: str, lastname: str, domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email Finder"
    
    """
    url = f"https://gabriel-cian-email-finder-v1.p.rapidapi.com/find-mail?api_key=kpIkHp5bk8gTrXx2MKx8&firstname={firstname}&lastname={lastname}&domain={domain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gabriel-cian-email-finder-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

