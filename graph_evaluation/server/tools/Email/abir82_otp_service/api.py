import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_otp(email_to: str, email_from: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "On calling this Endpoint the OTP will be sent from the user Email to the destination Email"
    
    """
    url = f"https://abir82-otp-service.p.rapidapi.com/"
    querystring = {'email_to': email_to, 'email_from': email_from, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abir82-otp-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

