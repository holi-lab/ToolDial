import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historical_location_vessel_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical Location API allows you to choose a specific location and a timestamp and see all vessels that pass through your selected zone in the selected timeframe."
    
    """
    url = f"https://maritime-ships-and-ports-database.p.rapidapi.com/api/v0/report"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maritime-ships-and-ports-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_vessel_data_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The historical vessel data API allows you to see vessel location and other events throughout your chosen dates. You can access the data easily and see vessel historical movement, activities speed, course, heading, destination and more."
    
    """
    url = f"https://maritime-ships-and-ports-database.p.rapidapi.com/api/v0/vessel_history?"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maritime-ships-and-ports-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_traffic_tracking_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Location Traffic Tracking API allows you to scan an area in the sea or ocean to see all the ships in your selected area. This API Endpoint lets you find and track all the vessels in the radius and monitor maritime traffic in your zone."
    
    """
    url = f"https://maritime-ships-and-ports-database.p.rapidapi.com/api/v0/vessel_inradius"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maritime-ships-and-ports-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vessel_finder_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can search and withdraw all vessels by type, similar names, draught, deadweight, year built, and more. You don't need to know vessel MMSI or IMO for this endpoint, just filter the search criteria and receive the API response with all vessels matching your parameters."
    
    """
    url = f"https://maritime-ships-and-ports-database.p.rapidapi.com/api/v0/vessel_find?"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maritime-ships-and-ports-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def port_finder_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find maritime ports information, location, time zone, country even when you dont know the exact port name."
    
    """
    url = f"https://maritime-ships-and-ports-database.p.rapidapi.com/api/v0/port_find?"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maritime-ships-and-ports-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vessel_info_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vessel Detailed Information"
    
    """
    url = f"https://maritime-ships-and-ports-database.p.rapidapi.com/api/v0/vessel_info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maritime-ships-and-ports-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_ship_tracking_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datalastic API provides ship’s current position and related information such as longitude, altitude, speed, vessel course, vessel destination, and more."
    
    """
    url = f"https://maritime-ships-and-ports-database.p.rapidapi.com/api/v0/vessel?"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maritime-ships-and-ports-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

