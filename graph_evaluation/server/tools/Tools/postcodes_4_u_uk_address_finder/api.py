import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_postcode_xml(postcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Royal Mail PAF database for list of addresses based on postcode (XML)"
    postcode: Search postcode
        
    """
    url = f"https://samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com/ByPostcode/xml"
    querystring = {'postcode': postcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_postcode_json(postcode: str, callback: str='return', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Royal Mail PAF database for a list of addresses based on a postcode, get results in JSON format"
    postcode: Search postcode
        callback: name of your javascript callback funtion
        
    """
    url = f"https://samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com/ByPostcode/json"
    querystring = {'postcode': postcode, }
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_address_id_xml(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Royal Mail PAF database by address id"
    id: Postcodes4U AddressId
        
    """
    url = f"https://samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com/byid/xml"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_address_id_json(is_id: int, callback: str='return', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the PAF Royal Mail database for an address based on address id"
    id: Postcodes4U Address Id
        callback: the name of your javascript callback function
        
    """
    url = f"https://samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com/byid/json"
    querystring = {'id': is_id, }
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "samsinfield-postcodes-4-u-uk-address-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

