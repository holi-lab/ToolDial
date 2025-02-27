import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def is_a_business(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests to this endpoint will return `true` if the number is a **Whatsapp for Business** account, or `false` if it's not."
    phone: The whatsapp number must be written as: `countrycode` and `number`; do NOT include any non-number character, spaces, or anything which is not a number.
Examples: of correct numbers are: `34123456789` (for spain) or `491234567890` (for Germany).
        
    """
    url = f"https://bulk-whatsapp-validator.p.rapidapi.com/isbiz"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-whatsapp-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_whatsapp_number(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the phone number you want to validate.
		Learn how to use it on this [video](https://youtu.be/_h5ybzp9UNU)."
    phone: The whatsapp number must be written as: number (including countrycode); do **NOT** include any non-number character, spaces, or anything which is not a number.
Examples: of correct numbers are: 34123456789 (for spain) or 491234567890 (for Germany).
        
    """
    url = f"https://bulk-whatsapp-validator.p.rapidapi.com/wchk"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-whatsapp-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about_info(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the *About* state of the WA number on the query."
    
    """
    url = f"https://bulk-whatsapp-validator.p.rapidapi.com/about"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-whatsapp-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about_info_free(phone: int, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You need a **valid token** to use this endpoint, learn how to get it on this [video](https://youtu.be/TcaAwKpAkl8). 
		This endpoint will return the `About` info of the number, and the time when the value was set."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        
    """
    url = f"https://bulk-whatsapp-validator.p.rapidapi.com/free/about"
    querystring = {'phone': phone, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-whatsapp-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_a_business_free(phone: int, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You need a **valid token** to use this endpoint, learn how to get it on this [video](https://youtu.be/TcaAwKpAkl8). 
		Requests to this endpoint will return `true` if the number is a **Whatsapp for Business** account, or `false` if it's not."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        
    """
    url = f"https://bulk-whatsapp-validator.p.rapidapi.com/free/isbiz"
    querystring = {'phone': phone, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-whatsapp-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_wa_number_free(phone: int, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You need a **valid token** to use this endpoint, learn how to get it on this [video](https://youtu.be/TcaAwKpAkl8). 
		The response is `true` if the phone is registered on whatsapp."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        
    """
    url = f"https://bulk-whatsapp-validator.p.rapidapi.com/free/wchk"
    querystring = {'phone': phone, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-whatsapp-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

