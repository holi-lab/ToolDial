import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def android_iconset_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate iconset for Android devices."
    url: The url of image to be iconified.
        
    """
    url = f"https://marcinpraski-icongenerator-v1.p.rapidapi.com/android/?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcinpraski-icongenerator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def universal_iconset_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate universal iconset for iOS 6.1+ and newest devices."
    url: The url of image to be iconified.
        
    """
    url = f"https://marcinpraski-icongenerator-v1.p.rapidapi.com/universal/?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcinpraski-icongenerator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def old_iphone_iconset_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate iconset suitable for projects aimed at iPhones running iOS 6.1 and below."
    url: The url of image to be iconified.
        
    """
    url = f"https://marcinpraski-icongenerator-v1.p.rapidapi.com/iphoneold/?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcinpraski-icongenerator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ipad_iconset_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate iconset suitable for projects aimed at iOS 6.1+ and newest iPads."
    url: The url of image to be iconified.
        
    """
    url = f"https://marcinpraski-icongenerator-v1.p.rapidapi.com/ipad/?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcinpraski-icongenerator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def old_ipad_iconset_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate iconset suitable for iPads running iOS 6.1 and below."
    url: The url of image to be iconified.
        
    """
    url = f"https://marcinpraski-icongenerator-v1.p.rapidapi.com/ipadold/?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcinpraski-icongenerator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def old_universal_iconset_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate universal iconset for devices running iOS 6.1 and below."
    url: The url of image to be iconified.
        
    """
    url = f"https://marcinpraski-icongenerator-v1.p.rapidapi.com/universalold/?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcinpraski-icongenerator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iphone_iconset_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate iconset suitable for projects aimed at iOS 6.1+ and newest iPhones."
    url: The url of image to be iconified.
        
    """
    url = f"https://marcinpraski-icongenerator-v1.p.rapidapi.com/iphone/?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcinpraski-icongenerator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

