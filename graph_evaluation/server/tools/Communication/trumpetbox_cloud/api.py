import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def messages_getsentmessagesfromaccount(key: str, priority: str, limit: str, api: str, device: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    key: TrumpetBox Cloud API KEY
        priority: Only get prioritized sent messages (Optional)
1 = Yes
0 = No (Default)
        limit: Number of results to return, default is 10 (Optional)


        api: Only get sent messages by API (Optional)
1 = Yes
0 = No (Default)
        device: Get messages only from specific device (Optional)
        page: Pagination number to navigate result sets (Optional)


        
    """
    url = f"https://trumpetbox-cloud.p.rapidapi.com/sent"
    querystring = {'key': key, 'priority': priority, 'limit': limit, 'api': api, 'device': device, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trumpetbox-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressbook_getcontactsfromaccount(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    key: Your TrumpetBox Cloud API KEY
        
    """
    url = f"https://trumpetbox-cloud.p.rapidapi.com/contacts"
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trumpetbox-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_getreceivedmessagesfromaccount(key: str, limit: str, page: str, device: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    key: TrumpetBox Cloud API KEY
        limit: Number of results to return, default is 10 (Optional)
        page: Pagination number to navigate result sets (Optional)
        device: Get received messages from specific device (Optional)
        
    """
    url = f"https://trumpetbox-cloud.p.rapidapi.com/received"
    querystring = {'key': key, 'limit': limit, 'page': page, 'device': device, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trumpetbox-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def devices_getalldeviceinfofromaccount(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    key: TrumpetBox Cloud API KEY
        
    """
    url = f"https://trumpetbox-cloud.p.rapidapi.com/devices"
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trumpetbox-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def devices_getasingledeviceinfofromaccount(is_id: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: ID of the device
        key: TrumpetBox Cloud API KEY
        
    """
    url = f"https://trumpetbox-cloud.p.rapidapi.com/device"
    querystring = {'id': is_id, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trumpetbox-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_getpendingmessagesfromaccount(device: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    device: ID of the specific device you want to get pending messages (Optional)


        key: TrumpetBox Cloud API KEY
        
    """
    url = f"https://trumpetbox-cloud.p.rapidapi.com/pending"
    querystring = {'device': device, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trumpetbox-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressbook_getgroupsfromaccount(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    key: Your TrumpetBox Cloud API KEY
        
    """
    url = f"https://trumpetbox-cloud.p.rapidapi.com/groups"
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trumpetbox-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

