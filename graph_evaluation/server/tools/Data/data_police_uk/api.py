import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def last_updated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crime data in the API is updated once a month. Find out when it was last updated."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/crime-last-updated"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crime_categories(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of valid categories for a given data set date."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/crime-categories"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def street_level_outcomes(date: str, lat: str, lng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Outcomes at street-level; either at a specific location, within a 1 mile radius of a single point, or within a custom area."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/outcomes-at-location"
    querystring = {'date': date, 'lat': lat, 'lng': lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_force(force: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides detailed information about a specific force."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/forces/{Force}"
    querystring = {'force': force, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crimes_at_location(date: str, lat: str, lng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns just the crimes which occurred at the specified location, rather than those within a radius. If given latitude and longitude, finds the nearest pre-defined location and returns the crimes which occurred there."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/crimes-at-location"
    querystring = {'date': date, 'lat': lat, 'lng': lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def street_level_crimes(lat: str, lng: str, date: str='2013-06', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crimes at street-level; either within a 1 mile radius of a single point, or within a custom area. For custom areas see http://data.police.uk/docs/method/crime-street/"
    lat: Latitude
        lng: Longitude
        date: Month to show in YYYY-MM format
        
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/crimes-street/all-crime"
    querystring = {'lat': lat, 'lng': lng, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def outcomes_for_a_specific_crime(persistent_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the outcomes (case history) for the specified crime. Crime ID is 64-character identifier, as returned by other API methods."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/outcomes-for-crime/{Persistent_ID}"
    querystring = {'persistent_id': persistent_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def street_level_availability(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of available data sets."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/crimes-street-dates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crimes_with_no_location(date: str, force: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of crimes where the responsible force hasn't specified a location."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/crimes-no-location"
    querystring = {'date': date, 'force': force, 'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forces(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all the police forces available via the API. Unique force identifiers obtained here are used in requests for force-specific data via other methods."
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/forces"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def local_neighbourhood(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/locate-neighbourhood"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(county: str, neighbourhood: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://stolenbikes88-datapoliceuk.p.rapidapi.com/{county}/{Neighbourhood}/events"
    querystring = {'neighbourhood': neighbourhood, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stolenbikes88-datapoliceuk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

