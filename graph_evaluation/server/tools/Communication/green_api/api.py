import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getoutgoingmessagesjournal(wainstanceidinstance: str, apitokeninstance: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Документация [LastOutgoingMessages](https://green-api.com/docs/api/journals/LastOutgoingMessages/)"
    
    """
    url = f"https://green-api2.p.rapidapi.com/waInstance{idInstance}/lastOutgoingMessages/{apitokeninstance}"
    querystring = {'waInstanceidInstance': wainstanceidinstance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "green-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def logoutaccount(wainstanceidinstance: str, apitokeninstance: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Документация [Logout](https://green-api.com/docs/api/account/Logout/)"
    
    """
    url = f"https://green-api2.p.rapidapi.com/waInstance{idInstance}/logout/{apitokeninstance}"
    querystring = {'waInstanceidInstance': wainstanceidinstance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "green-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfilefromincomingmessage(wainstanceidinstance: str, idmessage: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Документация [DownloadFile](https://green-api.com/docs/api/receiving/files/DownloadFile/)"
    
    """
    url = f"https://green-api2.p.rapidapi.com/waInstance{idInstance}/downloadFile/{idmessage}"
    querystring = {'waInstanceidInstance': wainstanceidinstance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "green-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getqrcode(wainstanceidinstance: str, apitokeninstance: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Документация [QR](https://green-api.com/docs/api/account/QR/)"
    
    """
    url = f"https://green-api2.p.rapidapi.com/waInstance{idInstance}/qr/{apitokeninstance}"
    querystring = {'waInstanceidInstance': wainstanceidinstance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "green-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchats(wainstanceidinstance: str, apitokeninstance: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Документация [GetContacts](https://green-api.com/docs/api/service/GetContacts/)"
    
    """
    url = f"https://green-api2.p.rapidapi.com/waInstance{idInstance}/getChats/{apitokeninstance}"
    querystring = {'waInstanceidInstance': wainstanceidinstance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "green-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rebootaccount(wainstanceidinstance: str, apitokeninstance: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Документация [Reboot](https://green-api.com/docs/api/account/Reboot/)"
    
    """
    url = f"https://green-api2.p.rapidapi.com/waInstance{idInstance}/reboot/{apitokeninstance}"
    querystring = {'waInstanceidInstance': wainstanceidinstance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "green-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def receivenotification(wainstanceidinstance: str, apitokeninstance: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Документация [ReceiveNotification](https://green-api.com/docs/api/receiving/technology-http-api/ReceiveNotification/)"
    
    """
    url = f"https://green-api2.p.rapidapi.com/waInstance{idInstance}/receiveNotification/{apitokeninstance}"
    querystring = {'waInstanceidInstance': wainstanceidinstance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "green-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

