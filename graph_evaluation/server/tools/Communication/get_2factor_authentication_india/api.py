import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def block_number_sms_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to add number to SMS Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/SMS/ADD"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def un_block_number_sms_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to remove number from SMS Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/SMS/REMOVE"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def un_block_number_voice_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to remove number from VOICE Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: Get one from http://2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/VOICE/REMOVE"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_number_voice_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to add number to VOICE Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: Get one from http://2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/VOICE/ADD"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

