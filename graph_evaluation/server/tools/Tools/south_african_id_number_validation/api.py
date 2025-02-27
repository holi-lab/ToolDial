import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def id_to_validator(is_id: str, gender: str='male', dob: str='890712', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will validate a South African ID number"
    
    """
    url = f"https://south-african-id-number-validation.p.rapidapi.com/{is_id}"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    if dob:
        querystring['dob'] = dob
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "south-african-id-number-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

