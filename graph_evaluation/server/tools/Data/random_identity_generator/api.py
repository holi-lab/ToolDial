import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_random_identity_default_format(gender: str='male', minage: str='18', maxage: str='65', country: str='us', nameset: str='american', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random identity base on the parametes/query provided."
    
    """
    url = f"https://random-identity-generator.p.rapidapi.com/identity"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    if minage:
        querystring['minAge'] = minage
    if maxage:
        querystring['maxAge'] = maxage
    if country:
        querystring['country'] = country
    if nameset:
        querystring['nameSet'] = nameset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-identity-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

