import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_profile_picture(phone: int, quality: str=None, pictype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the whatsapp number profile picture, you can select its resolution (`high` or `low`) and the response format: `png` file, `url` or `base64` encoded file."
    
    """
    url = f"https://whatsapp-data.p.rapidapi.com/wspicture"
    querystring = {'phone': phone, }
    if quality:
        querystring['quality'] = quality
    if pictype:
        querystring['pictype'] = pictype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_business_data(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint will provide: `description`, `website`, `email`, `business hours`, `address` and `category`; if the number is a whatsapp for business account."
    
    """
    url = f"https://whatsapp-data.p.rapidapi.com/bizinfo"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-data.p.rapidapi.com"
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
    
    """
    url = f"https://whatsapp-data.p.rapidapi.com/isbiz"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the `About` data of the Whatsapp for business number, and when the value was set."
    
    """
    url = f"https://whatsapp-data.p.rapidapi.com/about"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-data.p.rapidapi.com"
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
    
    """
    url = f"https://whatsapp-data.p.rapidapi.com/wchk"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

