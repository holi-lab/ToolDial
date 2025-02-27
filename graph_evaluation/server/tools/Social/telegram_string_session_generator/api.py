import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def submit_code_telegram_submitcode_get(request_id: str, otp: int, password: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Submit the Telegram code and password to complete login and generate string session.
		
		- **request_id**: The request_id received from the /sendCode request.
		- **code**: The code that Telegram sent. Note that if you have sent this
		            code through the application itself it will immediately
		            expire. If you want to send the code, obfuscate it somehow.
		            If you're not doing any of this you can ignore this note.
		- **password**: 2FA password, id no password is set leave it empty."
    
    """
    url = f"https://telegram-string-session-generator.p.rapidapi.com/telegram/submitCode/"
    querystring = {'request_id': request_id, 'otp': otp, }
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram-string-session-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_time_code_telegram_getcode_get(app_id: int, app_hash: str, phone_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sends the Telegram code needed to login to the given phone number.
		
		- **app_id**: The API ID you obtained from https://my.telegram.org.
		- **app_hash**: The API hash you obtained from https://my.telegram.org.
		- **phone_number**: The phone to which the code will be sent."
    
    """
    url = f"https://telegram-string-session-generator.p.rapidapi.com/telegram/getCode/"
    querystring = {'app_id': app_id, 'app_hash': app_hash, 'phone_number': phone_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram-string-session-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_login_status_telegram_status_get(request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the status of a request.
		
		- **request_id**: The request_id received from the /sendCode request."
    
    """
    url = f"https://telegram-string-session-generator.p.rapidapi.com/telegram/status/"
    querystring = {'request_id': request_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram-string-session-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

