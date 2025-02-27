import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_variable_value(appvariable: str, appkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This APi is used to Retrieve the variable value stored in the store"
    appvariable: AppVariable Value
        appkey: AppKey Value
        
    """
    url = f"https://online-key-value-store.p.rapidapi.com/api/KeyVal/GetValue/{appkey}/{appvariable}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-key-value-store.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_app_key(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Api to Get Your App Key"
    
    """
    url = f"https://online-key-value-store.p.rapidapi.com/api/KeyVal/GetAppKey"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-key-value-store.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

