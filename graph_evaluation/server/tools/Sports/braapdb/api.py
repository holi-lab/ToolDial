import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def experience_level(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: Experience Level Id to get
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/ExperienceLevel/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_parks(experiencelevel: str=None, terrainhas: str=None, isopen: bool=None, allowedvehicles: str=None, namecontains: str=None, dist: str=None, lng: str=None, lat: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Parks. Max of 10 results"
    experiencelevel: The Experience Level for the park. Valid inputs are: beginner, intermediate, advanced, or expert
        terrainhas: Park has Terrain Type. Valid inputs are: RhythmSection, Roots, Mud, Sand, Jumps, HillClimbs, HardPack, Gravle, LargeLooseRocks, OtherHazards
        isopen: If the park is open or not
        allowedvehicles: Type of Vehicle Allowed. Valid inputs are Motorcycles, Jeep, SxS, ATV, or Snowmobile
        namecontains: Park Name Contains. Note: Must be min of 3 chars
        dist: Radius around Lat Lng to search. Default is 25 Miles. If used, Lng and Lat must also be used
        lng: Longitude of where to start the search. Decimal degrees -180 to 180. If used, Lat must also be used
        lat: Latitude of where to start the search. Decimal degrees -90 to 90 If used, Lng must also be used
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/Parks"
    querystring = {}
    if experiencelevel:
        querystring['ExperienceLevel'] = experiencelevel
    if terrainhas:
        querystring['TerrainHas'] = terrainhas
    if isopen:
        querystring['IsOpen'] = isopen
    if allowedvehicles:
        querystring['allowedvehicles'] = allowedvehicles
    if namecontains:
        querystring['namecontains'] = namecontains
    if dist:
        querystring['dist'] = dist
    if lng:
        querystring['lng'] = lng
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trail by ID"
    
    """
    url = f"https://brappdbv2.p.rapidapi.com/Trail/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ride_report(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Ride Report by ID"
    
    """
    url = f"https://brappdbv2.p.rapidapi.com/RideReport/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def info_items_for_park(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the Info Items for this Park"
    
    """
    url = f"https://brappdbv2.p.rapidapi.com/Park/{is_id}/InfoItems"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def info_item(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an Info Item by ID"
    id: Info item ID to get
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/InfoItem/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ride_reports_for_park(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the Ride Reports for this Park"
    
    """
    url = f"https://brappdbv2.p.rapidapi.com/Park/{is_id}/RideReports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def park(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Park By ID"
    id: A Park ID. Example 250
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/Park/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def permitted_vehicles(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Permitted Vehicles"
    id: Permitted Vehicles ID to get
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/PermittedVehicles/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_info_items_for_trail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Info Items For a Trail"
    
    """
    url = f"https://brappdbv2.p.rapidapi.com/Trail/{is_id}/InfoItems"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def terrain(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Terrain Object by ID"
    id: ID of the Terrain object to get
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/Terrain/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def infolink(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get InfoLink by ID"
    id: ID of the InfoLink to get
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/InfoLink/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gpspoint(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get GPSPoint by ID"
    id: ID of the GPSPoint to get
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/GPSPoint/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trails_for_park(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the Trails for this Park"
    id: Park ID to get trails for
        
    """
    url = f"https://brappdbv2.p.rapidapi.com/Park/{is_id}/Trails"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brappdbv2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

