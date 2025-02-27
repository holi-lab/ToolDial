import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getinfo(subaccount: str=None, returnformat: str='xml or json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on an account"
    subaccount: subaccount to get info for
        returnformat: return type
        
    """
    url = f"https://allmysms-v9.p.rapidapi.com/getInfo"
    querystring = {}
    if subaccount:
        querystring['subAccount'] = subaccount
    if returnformat:
        querystring['returnformat'] = returnformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allmysms-v9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getblacklist(subaccount: str=None, returnformat: str='xml or json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all your blacklisted mobile phone numbers"
    subaccount: subaccount to get blacklist for
        returnformat: response format, xml or json
        
    """
    url = f"https://allmysms-v9.p.rapidapi.com/getBlacklist"
    querystring = {}
    if subaccount:
        querystring['subAccount'] = subaccount
    if returnformat:
        querystring['returnformat'] = returnformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allmysms-v9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpull(campid: str, returnformat: str='xml or json', subaccount: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recover all the answers (SMS-MO) related to a SMS campaign"
    campid: SMS campaign identifier
        returnformat: response format, xml or json
        subaccount: subaccount concerned by the request
        
    """
    url = f"https://allmysms-v9.p.rapidapi.com/getPull"
    querystring = {'campId': campid, }
    if returnformat:
        querystring['returnformat'] = returnformat
    if subaccount:
        querystring['subAccount'] = subaccount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allmysms-v9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubaccounts(returnformat: str='xml or json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the accounts attached to a master account"
    returnformat: response format, xml or json
        
    """
    url = f"https://allmysms-v9.p.rapidapi.com/getSubAccounts"
    querystring = {}
    if returnformat:
        querystring['returnformat'] = returnformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allmysms-v9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlists(returnformat: str='xml or json', subaccount: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the full list of the account's contacts list"
    returnformat: response format, xml or json
        subaccount: subaccount concerned by the request
        
    """
    url = f"https://allmysms-v9.p.rapidapi.com/getLists"
    querystring = {}
    if returnformat:
        querystring['returnformat'] = returnformat
    if subaccount:
        querystring['subAccount'] = subaccount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allmysms-v9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcontacts(listname: str, returnformat: str='xml or json', subaccount: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recover all the contacts stored in a contacts list"
    listname: Contacts list name
        returnformat: response format, xml or json
        subaccount: subaccount concerned by the request
        
    """
    url = f"https://allmysms-v9.p.rapidapi.com/getContacts"
    querystring = {'listName': listname, }
    if returnformat:
        querystring['returnformat'] = returnformat
    if subaccount:
        querystring['subAccount'] = subaccount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allmysms-v9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getacks(campid: str, returnformat: str='xml or json', subaccount: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "recover all operators acknowledgments"
    campid: campaign identifier
        returnformat: response format, xml or json
        subaccount: subaccount concerned by the request
        
    """
    url = f"https://allmysms-v9.p.rapidapi.com/getAcks"
    querystring = {'campId': campid, }
    if returnformat:
        querystring['returnformat'] = returnformat
    if subaccount:
        querystring['subAccount'] = subaccount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allmysms-v9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

