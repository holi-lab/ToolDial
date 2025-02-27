import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_data(phone: str='081223222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET detail profile information with phone Number
		change the  phone with your phone number want to search"
    
    """
    url = f"https://truecaller-data1.p.rapidapi.com/rapid_api/truecaller/{phone}"
    querystring = {}
    if phone:
        querystring['phone'] = phone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "truecaller-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

