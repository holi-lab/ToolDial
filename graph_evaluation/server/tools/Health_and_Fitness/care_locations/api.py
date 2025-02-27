import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def insurances(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all insurances and their policies"
    
    """
    url = f"https://care-locations.p.rapidapi.com/api/v1/insurances"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "care-locations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchtypes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the all searchable types. This may be practice types or specific treatments."
    
    """
    url = f"https://care-locations.p.rapidapi.com/api/v1/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "care-locations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insurancebyid(insuranceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves an insurance and its policies by its Id"
    
    """
    url = f"https://care-locations.p.rapidapi.com/api/v1/insurance/{insuranceid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "care-locations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(location: str, radius: int, type: str, insurancepolicyid: str='208a5f0c-60b1-4ffb-89f6-305582ea52d6', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for care providers within a certain radius around a place"
    location: The locationId, retrieved from the autocomplete endpoint
        radius: The radius (in km) to search within
        type: The type that should be filtered for, retrieved from as the Id from the types endpoint
        insurancepolicyid: If provided, filter results by insurance policies supported by the practice
        
    """
    url = f"https://care-locations.p.rapidapi.com/api/v1/search"
    querystring = {'location': location, 'radius': radius, 'type': type, }
    if insurancepolicyid:
        querystring['insurancePolicyId'] = insurancepolicyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "care-locations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the corresponding locationId for a given city, postcode or parts of it"
    value: Partial location value to execute the autocomplete for
        
    """
    url = f"https://care-locations.p.rapidapi.com/api/v1/autocomplete"
    querystring = {'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "care-locations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

