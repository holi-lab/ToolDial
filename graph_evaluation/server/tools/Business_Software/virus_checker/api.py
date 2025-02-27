import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def virus_scan(urladdress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Virus Scan accepts a File URL and checks the file for viruses"
    urladdress: This is the Files URL. Must be a publicly accessible address 
        
    """
    url = f"https://virus-checker.p.rapidapi.com/rapidapi/virusscan/rapidapitoken123456"
    querystring = {'urladdress': urladdress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "virus-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

