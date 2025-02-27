import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def carriers_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported carriers"
    
    """
    url = f"https://trackingmore.p.rapidapi.com/carriers/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackingmore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def packages_track_deprecated(carriercode: str, trackingnumber: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tracking information of specific package"
    carriercode: Carrier code gotten from .../carriers/list or carriers/detect endpoint
        trackingnumber: Tracking number of package
        lang: One of the following : en|cn|es|it|ru
        
    """
    url = f"https://trackingmore.p.rapidapi.com/packages/track"
    querystring = {'carrierCode': carriercode, 'trackingNumber': trackingnumber, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackingmore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def packages_v2_track(trackingnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tracking information of specific package"
    trackingnumber: Tracking number of package
        
    """
    url = f"https://trackingmore.p.rapidapi.com/packages/v2/track"
    querystring = {'trackingNumber': trackingnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackingmore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carriers_detect(trackingnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect carrier by providing tracking number"
    trackingnumber: The tracking number of parcel
        
    """
    url = f"https://trackingmore.p.rapidapi.com/carriers/detect"
    querystring = {'trackingNumber': trackingnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackingmore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

