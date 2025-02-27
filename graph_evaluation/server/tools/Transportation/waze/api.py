import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def autocomplete(q: str, lang: str='en', coordinates: str='37.376754, -122.023350', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Waze/Google autocomplete/type-ahead for places, locations and addresses."
    q: Free-text geographic query
        lang: The language of the results. See https://wazeopedia.waze.com/wiki/USA/Countries_and_Languages for a list of supported languages.
        coordinates: Geographic coordinates (latitude, longitude) bias. Highly recommended to use for getting accurate results.
        
    """
    url = f"https://waze.p.rapidapi.com/autocomplete"
    querystring = {'q': q, }
    if lang:
        querystring['lang'] = lang
    if coordinates:
        querystring['coordinates'] = coordinates
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "waze.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def driving_directions(source_coordinates: str, destination_coordinates: str, return_route_coordinates: bool=None, arrival_timestamp: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get routes and driving directions from Waze/Google."
    source_coordinates: Geographic coordinates (latitude, longitude pair) of the starting point
        destination_coordinates: Geographic coordinates (latitude, longitude pair) of the destination
        return_route_coordinates: Whether to return route coordinate pairs (`route_coordinates` field)
        arrival_timestamp: Unix-timestamp (seconds since epoch) of the arrival time (in case not specified directions will be returned for current time) 
        
    """
    url = f"https://waze.p.rapidapi.com/driving-directions"
    querystring = {'source_coordinates': source_coordinates, 'destination_coordinates': destination_coordinates, }
    if return_route_coordinates:
        querystring['return_route_coordinates'] = return_route_coordinates
    if arrival_timestamp:
        querystring['arrival_timestamp'] = arrival_timestamp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "waze.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_and_jams(top_right: str, bottom_left: str, max_alerts: int=20, max_jams: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get real-time alerts and jams from Waze in a geographic rectangular area defined by bottom-left and top-right latitude, longitude pairs."
    top_right: Top-right corner of the geographic rectangular area for which to get alerts and traffic jams. Specified as latitude, longitude pair.
        bottom_left: Bottom-left corner of the geographic rectangular area for which to get alerts and traffic jams. Specified as latitude, longitude pair.
        max_alerts: Maximum number of alerts to fetch (to avoid fetching alerts set it to `0`).
Default: `20`
        max_jams: Maximum number of traffic jams to fetch (to avoid fetching traffic jams, set it to `0`).
Default: `20`
        
    """
    url = f"https://waze.p.rapidapi.com/alerts-and-jams"
    querystring = {'top_right': top_right, 'bottom_left': bottom_left, }
    if max_alerts:
        querystring['max_alerts'] = max_alerts
    if max_jams:
        querystring['max_jams'] = max_jams
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "waze.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

