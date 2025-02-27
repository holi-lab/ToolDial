import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_places_by_latitude_and_longitude(latitude: str, longitude: str, group: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the places by latitude and longitude."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/places"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if group:
        querystring['group'] = group
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_timezones(page: str='1', name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the timezones."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/timezones"
    querystring = {}
    if page:
        querystring['page'] = page
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_continents(name: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the continents."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/continents"
    querystring = {}
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_countries(name: str=None, alpha3: str=None, alpha2: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the countries."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/countries"
    querystring = {}
    if name:
        querystring['name'] = name
    if alpha3:
        querystring['alpha3'] = alpha3
    if alpha2:
        querystring['alpha2'] = alpha2
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_states(name: str=None, page: str='1', abbreviation: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the states."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/states"
    querystring = {}
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    if abbreviation:
        querystring['abbreviation'] = abbreviation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_cities(name: str=None, statename: str=None, stateabbreviation: str=None, page: str='1', stateid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the cities."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/cities"
    querystring = {}
    if name:
        querystring['name'] = name
    if statename:
        querystring['stateName'] = statename
    if stateabbreviation:
        querystring['stateAbbreviation'] = stateabbreviation
    if page:
        querystring['page'] = page
    if stateid:
        querystring['stateId'] = stateid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_addresses_by_ip(ip: str='54.72.54.234', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the addresses by IP."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/ipAddresses"
    querystring = {}
    if ip:
        querystring['ip'] = ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_addresses_by_latitude_and_longitude(latitude: str, longitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the addresses by latitude and longitude."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/locations"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_addresses_by_postal_code(postalcode: str, countryalpha2: str=None, countryalpha3: str=None, countryname: str=None, countryid: str=None, stateid: str=None, statename: str=None, cityid: str=None, stateabbreviation: str=None, cityname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the addresses by postal code."
    
    """
    url = f"https://location74.p.rapidapi.com/v1/addresses"
    querystring = {'postalCode': postalcode, }
    if countryalpha2:
        querystring['countryAlpha2'] = countryalpha2
    if countryalpha3:
        querystring['countryAlpha3'] = countryalpha3
    if countryname:
        querystring['countryName'] = countryname
    if countryid:
        querystring['countryId'] = countryid
    if stateid:
        querystring['stateId'] = stateid
    if statename:
        querystring['stateName'] = statename
    if cityid:
        querystring['cityId'] = cityid
    if stateabbreviation:
        querystring['stateAbbreviation'] = stateabbreviation
    if cityname:
        querystring['cityName'] = cityname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

