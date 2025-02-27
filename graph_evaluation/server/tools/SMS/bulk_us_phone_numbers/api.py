import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_status(api: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if SMS is received"
    api: Set api=status to check if any SMS is received.
        
    """
    url = f"https://bulk-us-phone-numbers.p.rapidapi.com/USAPI.php"
    querystring = {'api': api, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-us-phone-numbers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_number_for_a_service(services: str, api: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get a new US number for a service/app."
    services: Select services=facebook for getting a number for facebook.
        api: Put api=new for getting a new US phone number.
        
    """
    url = f"https://bulk-us-phone-numbers.p.rapidapi.com/USAPI.php"
    querystring = {'services': services, 'api': api, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-us-phone-numbers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

