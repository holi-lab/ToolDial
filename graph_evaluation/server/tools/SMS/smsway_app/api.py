import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def devices_getdevice(is_id: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details about a specific registered device on your account"
    id: ID of the device
        key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/get/device?key="
    querystring = {'id': is_id, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressbook_deletecontact(is_id: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete saved contact number from your account"
    id: ID of contact number
        key: The api key you copied from the [SMS API KEY](https://smswayapp.com) (Tools -> API Keys) page
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/delete/contact?key="
    querystring = {'id': is_id, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressbook_getcontacts(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of your saved contacts"
    key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/get/contacts?key="
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address_book_getgroups(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of your cantact groups"
    key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/get/groups?key="
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_getpending(key: str, device: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of pending messages for sending"
    key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        device: Get messages only from specific device (Optional)
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/get/pending?key="
    querystring = {'key': key, 'device': device, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_getreceived(page: str, key: str, device: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of received messages on your account"
    page: Pagination number to navigate result sets (Optional)
        key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        device: Get messages only from specific device (Optional)
        limit: Number of results to return, default is 10 (Optional)
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/get/received?key="
    querystring = {'page': page, 'key': key, 'device': device, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_getsent(page: str, priority: str, api: str, device: str, limit: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of sent messages on your account"
    page: Pagination number to navigate result sets (Optional)
        priority: Only get prioritized sent messages (Optional)
1 = Yes
0 = No (Default)
        api: Only get sent messages by API (Optional)
1 = Yes
0 = No (Default)
        device: Get messages only from specific device (Optional)
        limit: Number of results to return, default is 10 (Optional)
        key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/get/sent?key="
    querystring = {'page': page, 'priority': priority, 'api': api, 'device': device, 'limit': limit, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressbook_deletegroup(key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details about a specific registered device on your account"
    key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        id: ID of contact group
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/delete/group?key="
    querystring = {'key': key, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def devices_getdevices(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of registered devices on your account"
    key: The api key you copied from the [SMSWAY APP](https://smswayapp.com) (Tools -> API Keys) page
        
    """
    url = f"https://smsway-app.p.rapidapi.com/api/get/devices?key="
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsway-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

