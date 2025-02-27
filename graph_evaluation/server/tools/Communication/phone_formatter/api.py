import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def regions(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of supported regions for the specified
		 IETF BCP 47 language tag string."
    language: Language in BCP 47 format
        
    """
    url = f"https://phone-formatter.p.rapidapi.com/regions/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-formatter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simpleregions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of supported regions. Method use "Accept-Language" header value for output list of
		 regions on specific language"
    
    """
    url = f"https://phone-formatter.p.rapidapi.com/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-formatter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple(phone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A simplified version of the formatting method that does not accept additional filtering and parsing parameters
		 and uses the language value from the "Accept-Language" header."
    phone: Phone number
        
    """
    url = f"https://phone-formatter.p.rapidapi.com/format"
    querystring = {}
    if phone:
        querystring['phone'] = phone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-formatter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

