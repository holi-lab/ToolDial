import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_verification_code(phone_number: str, verification_code: str, app_name: str='exampleapp', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check verification code we sent"
    phone_number: The phone number that sent the OTP
        verification_code: Verification code sent via SMS
        app_name: Same character string as when sending
        
    """
    url = f"https://wipple-sms-verify-otp.p.rapidapi.com/verify"
    querystring = {'phone_number': phone_number, 'verification_code': verification_code, }
    if app_name:
        querystring['app_name'] = app_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wipple-sms-verify-otp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

