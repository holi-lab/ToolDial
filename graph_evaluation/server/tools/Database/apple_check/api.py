import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def apple_warranty_basic_info_check(sn: str, imei: str, imei2: str='357232090625060', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "check information about Purchase Date, Activation Status, Warranty Status, Telephone Technical Support, Repairs and Service Coverage, Valid Purchase Date, AppleCare Eligible, Registered, Replaced and Loaner"
    
    """
    url = f"https://apple-check.p.rapidapi.com/coverage"
    querystring = {'sn': sn, 'imei': imei, }
    if imei2:
        querystring['imei2'] = imei2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iphone_ipad_sim_lock_status(imei: str, sn: str, imei2: str='357232090625060', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "check out if the device you would like to buy is locked or unlocked."
    
    """
    url = f"https://apple-check.p.rapidapi.com/simlock"
    querystring = {'imei': imei, 'sn': sn, }
    if imei2:
        querystring['imei2'] = imei2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iphone_ipad_carrier_sim_lock_status(imei: str, sn: str, imei2: str='357232090625060', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "check out if the device you would like to buy is locked or unlocked.
		check the carrier and country in which device is currently locked (network will not be given if the device is unlocked.)."
    
    """
    url = f"https://apple-check.p.rapidapi.com/carrier"
    querystring = {'imei': imei, 'sn': sn, }
    if imei2:
        querystring['imei2'] = imei2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ipad_wifi_only_icloud_lock_check(sn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "iPad iCloud Lock Check,only for wifi device.
		**Carrier device don't support!**"
    
    """
    url = f"https://apple-check.p.rapidapi.com/ipadcloud"
    querystring = {'sn': sn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

