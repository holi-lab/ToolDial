import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_by_fsa_code(postal_fsa: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query by Postal FSA ( example T6H )"
    postal_fsa: Query by value postal code FSA, example: \"A0A\"
        
    """
    url = f"https://vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com/rest/v1/public/boundary/ca/fsa"
    querystring = {'postal-fsa': postal_fsa, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_city_names_by_province_territory_name(province: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for City names by province/territory name"
    
    """
    url = f"https://vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com/rest/v1/public/boundary/ca/province/city/names"
    querystring = {'province': province, }
    if province:
        querystring['province'] = province
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_city_boundary_by_city_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for City Boundary by City name"
    
    """
    url = f"https://vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com/rest/v1/public/boundary/ca/province/city"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_all_province_territory_names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for All Province / Territory names"
    
    """
    url = f"https://vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com/rest/v1/public/boundary/ca/province/name"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_ldu_boundary_by_location_and_resolution(longitude: int, latitude: int, resolution: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for LDU H3 Boundary by Location and Resolution"
    
    """
    url = f"https://vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com/rest/v1/public/boundary/ca/h3/ldu/location/boundary"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if resolution:
        querystring['resolution'] = resolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_ldu_boundary_by_h3index(h3ndex: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Boundary by H3Index.
		Query for a LDU boundary by H3 Index, if a LDU Postal Code does not exist within the H3 Index Hexagon, an empty FeatureCollection is returned.
		
		**H3Index resolution must be greater than 8.**"
    
    """
    url = f"https://vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com/rest/v1/public/boundary/ca/h3/ldu/index/{h3ndex}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_ldu_boundary(postal_ldu: str, resolution: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query by a LDU Postal Code"
    postal_ldu: Query by LDU code.
        
    """
    url = f"https://vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com/rest/v1/public/boundary/ca/ldu"
    querystring = {'postal-ldu': postal_ldu, }
    if resolution:
        querystring['resolution'] = resolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-canada-postal-boundaries-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

