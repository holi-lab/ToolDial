import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_locations_by_vaccine_manufacturer(med_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the locations of available vaccines by a vaccine manufacturer query."
    
    """
    url = f"https://covid-19-live-vaccination-finder.p.rapidapi.com/locations/list-avail-vax-man"
    querystring = {'med_name': med_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-live-vaccination-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_locations_by_in_stock_status(in_stock: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get request that returns all locations that have the vaccine in stock."
    
    """
    url = f"https://covid-19-live-vaccination-finder.p.rapidapi.com/locations/list-in-stock"
    querystring = {'in_stock': in_stock, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-live-vaccination-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_locations_by_city_state(city: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get request that returns locations based off of a city and state query. City and State params are not case sensitive."
    
    """
    url = f"https://covid-19-live-vaccination-finder.p.rapidapi.com/locations/list-by-city-state"
    querystring = {'city': city, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-live-vaccination-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vaccine_locations_by_boundary(ne_longitude: str, sw_longitude: str, ne_latitude: str, sw_latitude: str, in_stock: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get request that takes NE and SW latitudes and longitudes as query params, then returns all locations within that boundary."
    
    """
    url = f"https://covid-19-live-vaccination-finder.p.rapidapi.com/locations/list-in-boundary"
    querystring = {'ne_longitude': ne_longitude, 'sw_longitude': sw_longitude, 'ne_latitude': ne_latitude, 'sw_latitude': sw_latitude, }
    if in_stock:
        querystring['in_stock'] = in_stock
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-live-vaccination-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vaccine_locations_by_zip(zip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get method that queries vaccine locations by zip"
    
    """
    url = f"https://covid-19-live-vaccination-finder.p.rapidapi.com/locations/list-by-zip"
    querystring = {'zip': zip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-live-vaccination-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vaccine_locations_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets vaccine locations by state"
    
    """
    url = f"https://covid-19-live-vaccination-finder.p.rapidapi.com/locations/list-by-state"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-live-vaccination-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_vaccine_locations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns back all vaccine locations and their details"
    
    """
    url = f"https://covid-19-live-vaccination-finder.p.rapidapi.com/locations/list-all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-live-vaccination-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

