import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def outdoors_legacy(lon: int=-105, radius: int=25, q_city_cont: str='Denver', q_state_cont: str='California', q_country_cont: str='Australia', limit: int=25, q_activities_activity_type_name_eq: str='hiking', lat: int=34, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Support for legacy queries. Data is no longer updated, but is still available."
    lon: Longitude
        radius: Used in combination with a lat/lon. Value is in miles.
        q_city_cont: Name of a city
        q_state_cont: US state or Canadian province
        q_country_cont: Name of a country
        limit: Default and maximum value is 100 results.
        q_activities_activity_type_name_eq: Limit results to specific outdoor activity [hiking, mountain biking, camping, caving, trail running, snow sports, atv, or horseback riding]
        lat: Latitude
        
    """
    url = f"https://trailapi-trailapi.p.rapidapi.com/activity/"
    querystring = {}
    if lon:
        querystring['lon'] = lon
    if radius:
        querystring['radius'] = radius
    if q_city_cont:
        querystring['q-city_cont'] = q_city_cont
    if q_state_cont:
        querystring['q-state_cont'] = q_state_cont
    if q_country_cont:
        querystring['q-country_cont'] = q_country_cont
    if limit:
        querystring['limit'] = limit
    if q_activities_activity_type_name_eq:
        querystring['q-activities_activity_type_name_eq'] = q_activities_activity_type_name_eq
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trailapi-trailapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_bike_trails(lat: int, lon: int, page: int=None, per_page: int=None, radius: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find trails near a point on the map."
    lat: Latitude
        lon: Longitude
        page: Results page (default: 1)
        per_page: Results per page (default: 50)
        radius: Radius (default: 25 miles; max: 100 miles)
        
    """
    url = f"https://trailapi-trailapi.p.rapidapi.com/trails/explore/"
    querystring = {'lat': lat, 'lon': lon, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trailapi-trailapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bike_trail_info(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about a single trail."
    
    """
    url = f"https://trailapi-trailapi.p.rapidapi.com/trails/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trailapi-trailapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trail_map_list(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a summary list of maps for this trail."
    
    """
    url = f"https://trailapi-trailapi.p.rapidapi.com/trails/{is_id}/maps/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trailapi-trailapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def map_gpx(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get GPX data for a map. Returns a file in GPX format. Input (id) is a map ID, not a trail ID. To get the map ID, call /maps with the trail ID."
    
    """
    url = f"https://trailapi-trailapi.p.rapidapi.com/trails/map/{is_id}/gpx/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trailapi-trailapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

