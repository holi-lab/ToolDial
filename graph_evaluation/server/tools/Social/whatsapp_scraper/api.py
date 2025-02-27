import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_business_info_get_free_token(phone: int, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free token needed, click [here](https://wa.me/34631428039?text=get-token) to get it. The endpoint will provide: `description`, `website`, `email`, `business hours`, `address` and `category`; if the number is a whatsapp for business account."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/free/bizinfo"
    querystring = {'phone': phone, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_whatsapp_for_business_get_free_token(token: str, phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free token needed, click [here](https://wa.me/34631428039?text=get-token) to get it. Requests to this endpoint will return `true` if the number is a **Whatsapp for Business** account, or `false` if it's not.
		Learn how to use this endpoint on this [video](https://youtu.be/txPQ4ROpfuc)."
    token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/free/isbiz"
    querystring = {'token': token, 'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about_get_free_token(token: str, phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free token needed, click [here](https://wa.me/34631428039?text=get-token) to get it. This endpoint will return the `About` info of the Whatsapp for business number, and when the value was set.
		Learn how to use this endpoint on this [video](https://youtu.be/txPQ4ROpfuc)."
    token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/free/about"
    querystring = {'token': token, 'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_registered_on_whatsapp_get_free_token(phone: int, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free token needed, click [here](https://wa.me/34631428039?text=get-token) to get it. This endpoint returns `true` if the phone is registered on whatsapp.
		Learn how to use this endpoint on this [video](https://youtu.be/txPQ4ROpfuc)."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/free/wchk"
    querystring = {'phone': phone, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the `About` info of the Whatsapp for business number, and when the value was set."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/about"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_profile_picture_get_free_token(phone: int, token: str, quality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free token needed, click [here](https://wa.me/34631428039?text=get-token) to get it. Gets the whatsapp number profile picture, you can select its resolution (`high` or `low`) and the response format: `png` file, `url` or `base64` encoded file.
		Learn how to use this endpoint on this [video](https://youtu.be/65eJN7S8sBw)."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        token: To get a free token, click [here](https://wa.me/34631428039?text=get-token) to send a whatsapp with the command `get-token`.
        quality: Quality of the Picture: High or Low.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/free/wspicture"
    querystring = {'phone': phone, 'token': token, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_profile_picture(phone: int, pictype: str=None, quality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the whatsapp number profile picture, you can select its resolution (`high` or `low`) and the response format: `png` file, `url` or `base64` encoded file."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        pictype: The type of response: url, jpg or base64
        quality: Quality of the Picture: High or Low.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/wspicture"
    querystring = {'phone': phone, }
    if pictype:
        querystring['pictype'] = pictype
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_whatsapp_for_business(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests to this endpoint will return `true` if the number is a **Whatsapp for Business** account, or `false` if it's not."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/isbiz"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_business_info(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint will provide: `description`, `website`, `email`, `business hours`, `address` and `category`; if the number is a whatsapp for business account."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/bizinfo"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_registered_on_whatsapp(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns `true` if the phone is registered on whatsapp."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        
    """
    url = f"https://whatsapp-scraper.p.rapidapi.com/wchk"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

