import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def uuidv4_generator(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate UUIDv4"
    
    """
    url = f"https://uuid-generator3.p.rapidapi.com/uuid/v4"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uuid-generator3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uuidv5_generator(name: str, namespace: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UUIDv5 generator"
    namespace: Allowed values: 

- dns
- url
- oid
- x500
- nil
        
    """
    url = f"https://uuid-generator3.p.rapidapi.com/uuid/v5"
    querystring = {'name': name, 'namespace': namespace, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uuid-generator3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uuidv3_generator(namespace: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UUIDv3 generator"
    namespace: Allowed values: 

- dns
- url
- oid
- x500
- nil
        
    """
    url = f"https://uuid-generator3.p.rapidapi.com/uuid/v3"
    querystring = {'namespace': namespace, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uuid-generator3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

