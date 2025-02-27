import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def orbit(number: int, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Orbital track for specified period"
    number: NORAD ID Number
        period: Orbital period (minutes)
        
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/{number}/orbit"
    querystring = {'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def launch_sites(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Launch sites around the world"
    
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/list/launch-sites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location(number: int, units: str=None, lat: int=47, lng: int=122, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current location by NORAD TLE number"
    number: NORAD TLE number
        units: Units for height and speed values.  Options are:
- imperial
- metric
        lat: Latitude to get satellite visibility
        lng: Longitude to get satellite visibility
        
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/{number}/location"
    querystring = {}
    if units:
        querystring['units'] = units
    if lat:
        querystring['lat'] = lat
    if lng:
        querystring['lng'] = lng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_satellites(days: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most popular satellites going back x days"
    days: Days to go back
        
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/top"
    querystring = {'days': days, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def visible_satellites(lat: str, lng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Satellites visible from a specific latitude and longitude"
    
    """
    url = f"https://uphere-space1.p.rapidapi.com/user/visible"
    querystring = {'lat': lat, 'lng': lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details(number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Details by Norad TLE number"
    number: Norad TLE number
        
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/{number}/details"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Categories used to filter satellites"
    
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/list/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Countries who have launched satellites which have been or are in orbit."
    
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/list/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def satellite_list(page: int, text: str='goes', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of satellites in orbit"
    page: Page of results (60 per page)
        text: Search by text
        country: Search by country
        
    """
    url = f"https://uphere-space1.p.rapidapi.com/satellite/list"
    querystring = {'page': page, }
    if text:
        querystring['text'] = text
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uphere-space1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

