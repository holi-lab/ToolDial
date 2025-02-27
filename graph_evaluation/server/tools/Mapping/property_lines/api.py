import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_us_state_boundaries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all US state boundaries."
    
    """
    url = f"https://property-lines.p.rapidapi.com/get_all_us_state_boundaries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "property-lines.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_radius_boundary(radius: int, state: str='ny', county: str='manhattan', lon: str='-73.94504387923337', lat: str='40.79975635358477', coords: str='40.79975635358477, -73.94504387923337', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the property boundaries associated with a given latitude and longitude coordinate within a given radius, if any exist."
    radius: Radius of property boundaries to return.
        state: (Optional) Provide state abbreviation and county to reduce response time.
        county: (Optional) Provide state abbreviation and county to reduce response time.
        lon: Provide either coords or lat & lon.
        lat: Provide either coords or lat & lon.
        coords: Provide coordinates (lat,lon)
        
    """
    url = f"https://property-lines.p.rapidapi.com/get_us_radius_property_boundaries"
    querystring = {'radius': radius, }
    if state:
        querystring['state'] = state
    if county:
        querystring['county'] = county
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    if coords:
        querystring['coords'] = coords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "property-lines.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_county_boundary(coords: str='40.79975635358477, -73.94504387923337', lat: str='40.79975635358477', lon: str='-73.94504387923337', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the  county property boundaries associated with a given latitude and longitude coordinate, if any exist."
    coords: Provide coordinates (lat,lon).
        lat: Provide either coords or lat & lon.
        lon: Provide either coords or lat & lon.
        
    """
    url = f"https://property-lines.p.rapidapi.com/get_us_county_boundary"
    querystring = {}
    if coords:
        querystring['coords'] = coords
    if lat:
        querystring['lat'] = lat
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "property-lines.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_counties_in_state_boundaries(lon: str='-73.94504387923337', lat: str='40.79975635358477', state: str='il', coords: str='40.79975635358477, -73.94504387923337', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves all county boundaries in the state associated with a given latitude and longitude coordinate, if any exist."
    lon: Either coords or lat & lon are required.
        lat: Either coords or lat & lon are required. 
        state: (Optional) Provide a state abbreviation to reduce response time.
        
    """
    url = f"https://property-lines.p.rapidapi.com/get_all_counties_for_us_state"
    querystring = {}
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    if state:
        querystring['state'] = state
    if coords:
        querystring['coords'] = coords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "property-lines.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_single_boundary(state: str=None, county: str=None, lon: str='-73.94531505009626', coords: str='40.79982062892406, -73.94531505009626', lat: str='40.79982062892406,', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the property boundaries associated with a given latitude and longitude coordinate, if any exist."
    state: (Optional) Provide state abbreviation and county to reduce response time.
        county: (Optional) Provide state and county to reduce response time.
        lon: Either coords or lat & lon are required.
        coords: Provide coordinates (lat,lon).
        lat: Either coords or lat & lon are required. 
        
    """
    url = f"https://property-lines.p.rapidapi.com/get_single_us_boundary"
    querystring = {}
    if state:
        querystring['state'] = state
    if county:
        querystring['county'] = county
    if lon:
        querystring['lon'] = lon
    if coords:
        querystring['coords'] = coords
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "property-lines.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_state_boundary(lon: str='-73.94504387923337', lat: str='40.79975635358477', coords: str='40.79975635358477, -73.94504387923337', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the  state property boundaries associated with a given latitude and longitude coordinate, if any exist."
    lat: Provide either coords or lat & lon.
        coords: Provide coordinates (lat,lon)
        
    """
    url = f"https://property-lines.p.rapidapi.com/get_us_state_boundary"
    querystring = {}
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    if coords:
        querystring['coords'] = coords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "property-lines.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

