import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recent_earthquakes(interval: str, count: str='100', radius: str='1000', units: str='miles', intensity: str='1', magnitude: str='3', longitude: str='-118.3706975', start: str='1', latitude: str='33.962523', type: str='earthquake', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 100 most recent earthquakes from all over in the world, from now to the date and time determined by the 'interval' property."
    interval: Get recent earthquakes from current time, back to a time based on the interval provided. Interval is an ISO8601 Duration, examples: P5D for the last 5 days, PT12H for the last 12 hours. https://en.wikipedia.org/wiki/ISO_8601#Durations
        count: Add to any call to limit the number of earthquakes returned.
100 earthquakes are returned by default, up to 1000 earthquakes can be returned using the 'count' parameter.
        radius: The length across half a circle of the search area. The latitude and longitude together make up the center of the circle, while the radius is the distance from the center to the outside of the circle. A 10 mile radius will be a circle 20 miles across with its center at the latitude/longitude point.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        units: Either miles or kilometers for the radius of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        intensity: Intensity can be added to limit results to only those earthquakes with at least the specified intensity or greater.
        magnitude: Magnitude can be added to limit results to only those earthquakes with at least the specified magnitude or greater.
        longitude: The longitude portion of the point at the center of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        start: Use the 'start' parameter to retrieve more than 1000 earthquakes, up to 1000 at a time. 'Start' is the offset to start at within the overall results.
100 earthquakes are returned by default, up to 1000 earthquakes can be returned using the 'count' parameter.
        latitude: The latitude portion of the point at the center of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        type: Type can be added to limit results to only those types specified, e.g.: earthquakes, quarry blasts, explosions. Use the 'types' endpoints to see all available types.
        
    """
    url = f"https://everyearthquake.p.rapidapi.com/recentEarthquakes"
    querystring = {'interval': interval, }
    if count:
        querystring['count'] = count
    if radius:
        querystring['radius'] = radius
    if units:
        querystring['units'] = units
    if intensity:
        querystring['intensity'] = intensity
    if magnitude:
        querystring['magnitude'] = magnitude
    if longitude:
        querystring['longitude'] = longitude
    if start:
        querystring['start'] = start
    if latitude:
        querystring['latitude'] = latitude
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m1_0_earthquakes_past_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 1.0+ Earthquakes, Past Day"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/1.0_day.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_earthquakes_past_30_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Earthquakes, Past Thirty Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/all_month.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m1_0_earthquakes_past_30_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 1.0+ Earthquakes, Past Thirty Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/1.0_month.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m2_5_earthquakes_past_30_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 2.5+ Earthquakes, Past Thirty Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/2.5_month.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m4_5_earthquakes_past_30_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 4.5+ Earthquakes, Past Thirty Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/4.5_month.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def significant_earthquakes_past_30_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Significant Earthquakes, Past Thirty Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/significant_month.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_earthquakes_past_7_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Earthquakes, Past Seven Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/all_week.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m1_0_earthquakes_past_7_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 1.0+ Earthquakes, Past Seven Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/1.0_week.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m2_5_earthquakes_past_7_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 2.5+ Earthquakes, Past Seven Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/2.5_week.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m4_5_earthquakes_past_7_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 4.5+ Earthquakes, Past Seven Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/4.5_week.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def significant_earthquakes_past_7_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Significant Earthquakes, Past Seven Days"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/significant_week.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m4_5_earthquakes_past_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 4.5+ Earthquakes, Past Day"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/4.5_day.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m2_5_earthquakes_past_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 2.5+ Earthquakes, Past Day"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/2.5_day.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m1_0_earthquakes_past_hour(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 1.0+ Earthquakes, Past Hour"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/1.0_hour.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_earthquakes_past_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Earthquakes, Past Day"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/all_day.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def significant_earthquakes_past_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Significant Earthquakes, Past Day"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/significant_day.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_earthquakes_past_hour(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Earthquakes, Past Hour"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/all_hour.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m2_5_earthquakes_past_hour(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 2.5+ Earthquakes, Past Hour"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/2.5_hour.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m4_5_earthquakes_past_hour(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Magnitude 4.5+ Earthquakes, Past Hour"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/4.5_hour.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def significant_earthquakes_past_hour(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Significant Earthquakes, Past Hour"
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/significant_hour.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_test(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple call to determine if the API is available and responding."
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/testPass"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earthquakes_by_date(startdate: str, enddate: str, intensity: str='1', longitude: str='-118.3706975', type: str='earthquake', count: str='100', latitude: str='33.962523', units: str='miles', start: str='1', radius: str='1000', magnitude: str='3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 100 most recent earthquakes from all over in the world, within a specified date range."
    startdate: The start date for the search period in YYYY-MM-DD format. All dates start and end at midnight UTC.
        enddate: The end date for the search period in YYYY-MM-DD format. All dates start and end at midnight UTC.
        intensity: Intensity can be added to limit results to only those earthquakes with at least the specified intensity or greater.
        longitude: The longitude portion of the point at the center of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        type: Type can be added to limit results to only those types specified, e.g.: earthquakes, quarry blasts, explosions. Use the 'types' endpoints to see all available types.
        count: Add to any call to limit the number of earthquakes returned.
100 earthquakes are returned by default, up to 1000 earthquakes can be returned using the 'count' parameter.
        latitude: The latitude portion of the point at the center of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        units: Either miles or kilometers for the radius of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        start: Use the 'start' parameter to retrieve more than 1000 earthquakes, up to 1000 at a time. 'Start' is the offset to start at within the overall results.
100 earthquakes are returned by default, up to 1000 earthquakes can be returned using the 'count' parameter.
        radius: The length across half a circle of the search area. The latitude and longitude together make up the center of the circle, while the radius is the distance from the center to the outside of the circle. A 10 mile radius will be a circle 20 miles across with its center at the latitude/longitude point.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        magnitude: Magnitude can be added to limit results to only those earthquakes with at least the specified magnitude or greater.
        
    """
    url = f"https://everyearthquake.p.rapidapi.com/earthquakesByDate"
    querystring = {'startDate': startdate, 'endDate': enddate, }
    if intensity:
        querystring['intensity'] = intensity
    if longitude:
        querystring['longitude'] = longitude
    if type:
        querystring['type'] = type
    if count:
        querystring['count'] = count
    if latitude:
        querystring['latitude'] = latitude
    if units:
        querystring['units'] = units
    if start:
        querystring['start'] = start
    if radius:
        querystring['radius'] = radius
    if magnitude:
        querystring['magnitude'] = magnitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all types of events ever recorded by the USGS, such as earthquakes, explosions, landslides and many more."
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earthquakes(units: str='miles', latitude: str='33.962523', intensity: str='1', type: str='earthquake', magnitude: str='3', radius: str='1000', count: str='100', longitude: str='-118.3706975', start: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 100 most recent earthquakes from all over in the world. No parameters required."
    units: Either miles or kilometers for the radius of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        latitude: The latitude portion of the point at the center of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        intensity: Intensity can be added to limit results to only those earthquakes with at least the specified intensity or greater.
        type: Type can be added to limit results to only those types specified, e.g.: earthquakes, quarry blasts, explosions. Use the 'types' endpoints to see all available types.
        magnitude: Magnitude can be added to limit results to only those earthquakes with at least the specified magnitude or greater.
        radius: The length across half a circle of the search area. The latitude and longitude together make up the center of the circle, while the radius is the distance from the center to the outside of the circle. A 10 mile radius will be a circle 20 miles across with its center at the latitude/longitude point.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        count: Add to any call to limit the number of earthquakes returned.
100 earthquakes are returned by default, up to 1000 earthquakes can be returned using the 'count' parameter.
        longitude: The longitude portion of the point at the center of the search circle.
Search for earthquakes near a certain location using latitude, longitude, radius and units. All four properties must be included when retrieving earthquakes near a certain location.
        start: Use the 'start' parameter to retrieve more than 1000 earthquakes, up to 1000 at a time. 'Start' is the offset to start at within the overall results.
100 earthquakes are returned by default, up to 1000 earthquakes can be returned using the 'count' parameter.
        
    """
    url = f"https://everyearthquake.p.rapidapi.com/earthquakes"
    querystring = {}
    if units:
        querystring['units'] = units
    if latitude:
        querystring['latitude'] = latitude
    if intensity:
        querystring['intensity'] = intensity
    if type:
        querystring['type'] = type
    if magnitude:
        querystring['magnitude'] = magnitude
    if radius:
        querystring['radius'] = radius
    if count:
        querystring['count'] = count
    if longitude:
        querystring['longitude'] = longitude
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_earthquake_near_me(latitude: str, longitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call this endpoint with your latitude and longitude and you'll receive a simple return for the most recent earthquake, within 100 miles, of the lat/long coordinates provided."
    
    """
    url = f"https://everyearthquake.p.rapidapi.com/latestEarthquakeNearMe"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everyearthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

