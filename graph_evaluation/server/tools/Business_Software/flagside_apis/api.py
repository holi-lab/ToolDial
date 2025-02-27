import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_feature_flag_value(secret: str, x_api_key: str, featureflagkey: str, authorization: str='Bearer {{authToken}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Optimized Endpoint to query the value of a feature Flag"
    
    """
    url = f"https://flagside-apis.p.rapidapi.com/ff/eval/{featureflagkey}"
    querystring = {'secret': secret, 'x-api-key': x_api_key, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flagside-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

