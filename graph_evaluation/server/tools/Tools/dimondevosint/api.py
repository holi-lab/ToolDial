import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def photo_from_given_number(phone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It searches for a photo in the internet about the phone number, if it matches, it returns the photo in bytes."
    
    """
    url = f"https://dimondevosint.p.rapidapi.com/photo"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dimondevosint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_number_information(phone: str, captchasid: str=None, vkhash: str=None, captcha: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It gives you some public information about the phone number."
    
    """
    url = f"https://dimondevosint.p.rapidapi.com/main"
    querystring = {'phone': phone, }
    if captchasid:
        querystring['captchaSID'] = captchasid
    if vkhash:
        querystring['vkHash'] = vkhash
    if captcha:
        querystring['captcha'] = captcha
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dimondevosint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inn_general_director(inn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the general director of a company by INN nymber."
    
    """
    url = f"https://dimondevosint.p.rapidapi.com/inn"
    querystring = {'inn': inn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dimondevosint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

