import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbycordsadv(radius: int, lat: int, lng: int, access_param: str='public', ev_connector_type_param: str='J1772', ev_network_param: str='Tesla,Tesla Destination', per_page: int=10, page: int=1, owner_type_param: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Uses Latitude & Longitude to get near by electric charging stations"
    radius: Radius in miles
        
    """
    url = f"https://electric-vehicle-charging-stations.p.rapidapi.com/getByCordsAdv"
    querystring = {'radius': radius, 'lat': lat, 'lng': lng, }
    if access_param:
        querystring['access_param'] = access_param
    if ev_connector_type_param:
        querystring['ev_connector_type_param'] = ev_connector_type_param
    if ev_network_param:
        querystring['ev_network_param'] = ev_network_param
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if owner_type_param:
        querystring['owner_type_param'] = owner_type_param
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccesstype(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns access type options for stations"
    
    """
    url = f"https://electric-vehicle-charging-stations.p.rapidapi.com/getAccessTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getevconnectors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return only electric charging connector types that can be used to filter GetByCordsAdv."
    
    """
    url = f"https://electric-vehicle-charging-stations.p.rapidapi.com/getConnectors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getowners(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return stations owned by the given types of owners. A single owner type, or a comma-separated list of multiple owner types, may be given."
    
    """
    url = f"https://electric-vehicle-charging-stations.p.rapidapi.com/getOwners"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnetworks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return only electric charging stations that belong to the given network. A single network, or a comma separated list of multiple networks, may be given."
    
    """
    url = f"https://electric-vehicle-charging-stations.p.rapidapi.com/getNetworks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbycords(radius: int, lng: int, lat: int, page: int=1, per_page: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Uses Latitude & Longitude to get near by electric charging stations"
    radius: Radius in miles
        
    """
    url = f"https://electric-vehicle-charging-stations.p.rapidapi.com/getByCords"
    querystring = {'radius': radius, 'lng': lng, 'lat': lat, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

