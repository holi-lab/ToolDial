import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_5_indonesia_domain_suggestion(name: str, action: str, n: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back 5 domain suggestion, available status and prices in IDR"
    name: fill this parameter with mandatory value of domain keyword, for example \"data\", \"online\", \"internet\", etc
        action: fill this parameter with mandatory value \"suggestdomain\"
        
    """
    url = f"https://indonesian-domain-name-suggestion-available.p.rapidapi.com/"
    querystring = {'name': name, 'action': action, }
    if n:
        querystring['n'] = n
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-domain-name-suggestion-available.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

