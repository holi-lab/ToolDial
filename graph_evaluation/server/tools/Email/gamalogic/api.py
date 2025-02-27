import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gamalogic_email_validation_api(emailid: str, api_key: str, speed_rank: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email Validation API"
    
    """
    url = f"https://gamalogic.p.rapidapi.com/gamalogic.com"
    querystring = {'emailid': emailid, 'api_key': api_key, }
    if speed_rank:
        querystring['speed_rank'] = speed_rank
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamalogic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

