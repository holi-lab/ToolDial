import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_for_hexagon_by_location_and_resolution(longitude: int, latitude: int, resolution: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Hexagon by Location and Resolution"
    
    """
    url = f"https://vanitysoft-uk-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/h3/location/boundary"
    querystring = {'longitude': longitude, 'latitude': latitude, 'resolution': resolution, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-uk-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_hexagon_by_h3index(h3index: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Hexagon by H3Index"
    
    """
    url = f"https://vanitysoft-uk-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/h3/index/{h3index}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-uk-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_uk_postal_sector_outline_boundaries(postal_sector: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "example: Query by "W41,W42" sectors."
    
    """
    url = f"https://vanitysoft-uk-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/uk/sectors"
    querystring = {'postal-sector': postal_sector, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-uk-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_uk_postal_district_outline_boundaries(postal_district: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "example: Query by "TW12" district"
    postal_district: Query by postal district code.
        
    """
    url = f"https://vanitysoft-uk-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/uk/districts"
    querystring = {'postal-district': postal_district, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-uk-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_postal_code_unit_boundary_h3_geo_boundary(postal_unit: str, resolution: int=13, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Postal Unit Boundary (ex. ZE1 0AE) 
		
		https://eng.uber.com/"
    postal_unit: Query by postal units. for example 'ZE1 0AE", or multiples.
        
    """
    url = f"https://vanitysoft-uk-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/uk"
    querystring = {'postal-unit': postal_unit, }
    if resolution:
        querystring['resolution'] = resolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-uk-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_postal_unit_boundary_by_h3index(h3ndex: int, resolution: int=13, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Postal Unit Boundary by H3Index
		
		If  a Postal Unit code exists within this hexagon a boundary is returned.
		adjusting the resolution(0-16) increases the hexagon area.
		
		https://eng.uber.com/"
    
    """
    url = f"https://vanitysoft-uk-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/uk/h3/unit/index/{h3ndex}"
    querystring = {}
    if resolution:
        querystring['resolution'] = resolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-uk-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_distance_between_two_h3_indexes(h3index1: str='8d2baad9c6f073f', h3index2: str='8d09a6b6ed2d37f', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Distance between two H3 Indexes"
    
    """
    url = f"https://vanitysoft-uk-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/h3/index1/{h3index1}/index2/{h3index2}/hex"
    querystring = {}
    if h3index1:
        querystring['h3Index1'] = h3index1
    if h3index2:
        querystring['h3Index2'] = h3index2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-uk-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

