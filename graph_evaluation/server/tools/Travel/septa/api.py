import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_alerts(req1: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://septa.p.rapidapi.com/hackathon/Alerts/get_alert_data.php"
    querystring = {}
    if req1:
        querystring['req1'] = req1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bus_trolley_schedules(req1: int, req2: int=17, req3: str='i', req6: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    req1: Stop ID
        req2: Route number
        req3: i for inbound, o for outbound
        req6: Number of results
        
    """
    url = f"https://septa.p.rapidapi.com/hackathon/BusSchedules/"
    querystring = {'req1': req1, }
    if req2:
        querystring['req2'] = req2
    if req3:
        querystring['req3'] = req3
    if req6:
        querystring['req6'] = req6
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bus_trolley_locations(route: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://septa.p.rapidapi.com/hackathon/TransitView/trips.php"
    querystring = {'route': route, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def next_to_arrive(req1: str, req2: str, req3: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a list of regional rail trains that will soon be traveling from point A to point B."
    req1: Origin station
        req2: Destination station
        req3: Number of results
        
    """
    url = f"https://septa.p.rapidapi.com/hackathon/NextToArrive/"
    querystring = {'req1': req1, 'req2': req2, }
    if req3:
        querystring['req3'] = req3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bus_detours(req1: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    req1: Route number
        
    """
    url = f"https://septa.p.rapidapi.com/hackathon/BusDetours/"
    querystring = {}
    if req1:
        querystring['req1'] = req1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regional_rail_schedules(req1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By train number, lol."
    
    """
    url = f"https://septa.p.rapidapi.com/hackathon/RRSchedules/"
    querystring = {'req1': req1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bus_trolley_routes(req1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    req1: Route number
        
    """
    url = f"https://septa.p.rapidapi.com/hackathon/Stops/"
    querystring = {'req1': req1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def plain_text_bus_schedules(req1: int, req2: str='17', req3: str='i', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Intended for SMS."
    req1: Stop ID
        req2: Route number
        req3: Inbound/outbound
        
    """
    url = f"https://septa.p.rapidapi.com/sms/"
    querystring = {'req1': req1, }
    if req2:
        querystring['req2'] = req2
    if req3:
        querystring['req3'] = req3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_locations(lon: int, lat: int, radius: int=3, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    lon: Longitude
        lat: Latitude
        radius: Number of miles (accepts a decimal)
        type: One of [bus_stops , rail_stations , perk_locations , trolley_stops , sales_locations]
        
    """
    url = f"https://septa.p.rapidapi.com/hackathon/locations/get_locations.php"
    querystring = {'lon': lon, 'lat': lat, }
    if radius:
        querystring['radius'] = radius
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainview(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Regional Rail real-time train locations"
    
    """
    url = f"https://septa.p.rapidapi.com/hackathon/TrainView/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter(req1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    req1: Number of posts
        
    """
    url = f"https://septa.p.rapidapi.com/hackathon/Twitter/"
    querystring = {'req1': req1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "septa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

