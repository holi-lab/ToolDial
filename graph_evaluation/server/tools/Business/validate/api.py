import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validate(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simply validate the email by passing it in the query parameter.
		The response json contains status key which will tell whether an email is valid or not.
		status key can have 3 values
		1) valid - indicates a valid email
		2) invalid - indicates an invalid email
		3) disposable - indicates disposable email
		The other keys are self explanatory"
    
    """
    url = f"https://validate3.p.rapidapi.com/v1/validate"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "validate3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

