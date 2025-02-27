import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def advanced_phone_number_lookup(dialcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Advanced Phone Number Information**
		
		*retrieves valuable technical data about a phone number*
		
		- validity
		
		- approximate location
		
		- timezone
		
		- carrier
		
		- line type
		
		- ported status
		
		- carrier
		
		- robocall/spam score
		
		- much more"
    
    """
    url = f"https://scout.p.rapidapi.com/v1/numbers/search"
    querystring = {'dialcode': dialcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scout.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basic_phone_number_validation(dialcode: str, country_code: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Basic Phone Number Validation**
		
		- determine if a phone number is in a valid format 
		
		- determine if a phone number is impossible for a particular region
		
		- properly format the number for various scenarios like international or local dialing
		
		- attempts to determine line type
		
		- determines country and country code"
    
    """
    url = f"https://scout.p.rapidapi.com/v1/numbers/verify"
    querystring = {'dialcode': dialcode, }
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scout.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

