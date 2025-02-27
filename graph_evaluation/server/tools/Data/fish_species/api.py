import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fish_api_group(meta_property: str, property_value: str, meta_property_attribute: str='class', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will group the available data by the <code>property_value</code> parameter. The available URL query parameters are <code>meta_property</code>,  <code>meta_property_attribute</code>, and <code>property_value</code>. The API will group and return all fishes that are matching the <code>property_value</code> that is given in the meta-object."
    meta_property: The property to search for
        property_value: The value of property that will be matched
        meta_property_attribute: The value of property that will be matched
        
    """
    url = f"https://fish-species.p.rapidapi.com/fish_api/group"
    querystring = {'meta_property': meta_property, 'property_value': property_value, }
    if meta_property_attribute:
        querystring['meta_property_attribute'] = meta_property_attribute
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fish-species.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fish_api_fishes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all available fishes that are available"
    
    """
    url = f"https://fish-species.p.rapidapi.com/fish_api/fishes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fish-species.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fish_api_fish_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return information for a specific fish"
    name: The fish to be found
        
    """
    url = f"https://fish-species.p.rapidapi.com/fish_api/fish/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fish-species.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

