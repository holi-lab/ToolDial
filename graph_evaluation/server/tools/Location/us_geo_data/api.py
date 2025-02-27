import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_zips_in_radius(lng: str='-121.89408', lat: str='37.29554', radius: str='5000', zip: str='95125', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter a zip code or latitude & longitude coordinates with a radius (in meters) to find other zip codes within the radius!"
    
    """
    url = f"https://us-geo-data2.p.rapidapi.com/zips/radius"
    querystring = {}
    if lng:
        querystring['lng'] = lng
    if lat:
        querystring['lat'] = lat
    if radius:
        querystring['radius'] = radius
    if zip:
        querystring['zip'] = zip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-geo-data2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_zip_codes(zip: str='95125,95126', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter one or more US zip codes for a basic zip code translation
		
		**For many zips, separate by comma**"
    
    """
    url = f"https://us-geo-data2.p.rapidapi.com/zips/lookup"
    querystring = {}
    if zip:
        querystring['zip'] = zip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-geo-data2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_zips_in_city(stateid: str, city: str, sortbypopulation: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter US city and State ID (ie "CA") to return all zip codes in a city.
		
		**Optional**: sortByPopulation "asc" or "desc""
    
    """
    url = f"https://us-geo-data2.p.rapidapi.com/zips/city"
    querystring = {'stateId': stateid, 'city': city, }
    if sortbypopulation:
        querystring['sortByPopulation'] = sortbypopulation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-geo-data2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_zips_in_county(county: str, stateid: str, sortbypopulation: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter US county and State ID (ie "CA") to return all zip codes in a city.
		
		**Optional**: sortByPopulation "asc" or "desc""
    
    """
    url = f"https://us-geo-data2.p.rapidapi.com/zips/county"
    querystring = {'county': county, 'stateId': stateid, }
    if sortbypopulation:
        querystring['sortByPopulation'] = sortbypopulation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-geo-data2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_zips_in_state(stateid: str, sortbypopulation: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter US State ID (ie "CA") to return all zip codes in a state.
		
		*Optional*: sortByPopulation "asc" or "desc""
    
    """
    url = f"https://us-geo-data2.p.rapidapi.com/zips/state"
    querystring = {'stateId': stateid, }
    if sortbypopulation:
        querystring['sortByPopulation'] = sortbypopulation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-geo-data2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

