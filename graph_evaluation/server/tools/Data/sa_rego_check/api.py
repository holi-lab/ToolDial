import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def registration_check(rego: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information returned from this check includes:
		
		Make
		Primary Colour
		Expiry Date 
		body or hull type
		compulsory third-party (CTP) insurer (vehicles only)
		heavy vehicle details (gross vehicle, combination, or trailer mass)."
    
    """
    url = f"https://sa-rego-check.p.rapidapi.com/check"
    querystring = {'rego': rego, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sa-rego-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

