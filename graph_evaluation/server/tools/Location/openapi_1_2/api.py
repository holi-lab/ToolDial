import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def agencies(format: str, callback: str='call', geo_area: str='35.80176,-78.64347|35.78061,-78.68218', agencies: str='12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource contains a list of agencies along with their properties. It may be restricted by a geographical area filter, to retrieve only a list of agencies in a particular area of interest. It may also be restricted by agencies' IDs."
    format: Format for the return value. One of ['json', 'jsonp'].
        callback: Callback function name for JSONP.
        geo_area: Geographical area filter (See the readme).
        agencies: A list of agency IDs which to retrieve, seperated by comma.
        
    """
    url = f"https://transloc-api-1-2.p.rapidapi.com/agencies.{format}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if geo_area:
        querystring['geo_area'] = geo_area
    if agencies:
        querystring['agencies'] = agencies
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transloc-api-1-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def arrival_estimates(agencies: str, format: str, routes: str='4000421,4000592,4005122', stops: str='4002123,4023414,4021521', callback: str='call', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource contains a list of arrival estimates, separated by stops. When accessing this resource you must specify either a list of agencies, a list of routes or a list of stops."
    agencies: A list of agency IDs which to retreive
        format: Format for the return value. One of ['json', 'jsonp'].
        routes: A list of route IDs for which to retrieve the arrival estimates.
        stops: A list of stop IDs for which to retrieve the arrival estimates.
        callback: Callback function name for JSONP.
        
    """
    url = f"https://transloc-api-1-2.p.rapidapi.com/arrival-estimates.{format}"
    querystring = {'agencies': agencies, }
    if routes:
        querystring['routes'] = routes
    if stops:
        querystring['stops'] = stops
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transloc-api-1-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def segments(agencies: str, format: str, routes: str='4000204', geo_area: str='35.80176,-78.64347|35.78061,-78.68218', callback: str='call', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource contains a list of all the segments that are required to visualize routes. It must be restricted by agencies' ID's. It may be restricted by a geographical area filter, to retreive only a list of segments in a particular area. It may also be restricted by route ID's."
    agencies: A list of agency IDs which to retrieve, seperated by comma.
        format: Format for the return value. One of ['json', 'jsonp'].
        routes: A list of route IDs which to retreive
        geo_area: Geographical area filter (See the readme).
        callback: Callback function name for JSONP.
        
    """
    url = f"https://transloc-api-1-2.p.rapidapi.com/segments.{format}"
    querystring = {'agencies': agencies, }
    if routes:
        querystring['routes'] = routes
    if geo_area:
        querystring['geo_area'] = geo_area
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transloc-api-1-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehicles(agencies: str, format: str, routes: str='4000421,4000592,4005122', geo_area: str='35.80176,-78.64347|35.78061,-78.68218', callback: str='call', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource contains a list of vehicle, their properties and their locations. It must be restricted by agencies' IDs. It may be further restricted by a geographical area filter, to retrieve only the list of routes in a particular area of interest. It may also be restricted by a list of route ID's."
    agencies: A list of agency IDs which to retreive
        format: Format for the return value. One of ['json', 'jsonp'].
        routes: A list of route IDs for which to retrieve the vehicles, separated by comma.
        geo_area: Geographical area filter (See the readme).
        
    """
    url = f"https://transloc-api-1-2.p.rapidapi.com/vehicles.{format}"
    querystring = {'agencies': agencies, }
    if routes:
        querystring['routes'] = routes
    if geo_area:
        querystring['geo_area'] = geo_area
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transloc-api-1-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def routes(agencies: str, format: str, geo_area: str='35.80176,-78.64347|35.78061,-78.68218', callback: str='call', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource contains a list of routes. It must be restricted by agencies' IDs. It may be restricted by a geographical area filter, to retrieve only a list of routes in a particular area."
    agencies: A list of agency IDs which to retrieve, seperated by comma.
        format: Format for the return value. One of ['json', 'jsonp'].
        geo_area: Geographical area filter (See the readme).
        callback: Callback function name for JSONP.
        
    """
    url = f"https://transloc-api-1-2.p.rapidapi.com/routes.{format}"
    querystring = {'agencies': agencies, }
    if geo_area:
        querystring['geo_area'] = geo_area
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transloc-api-1-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stops(agencies: str, format: str, geo_area: str='35.80176,-78.64347|35.78061,-78.68218', callback: str='call', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource contains a list of stops. It must be restricted by agencies' IDs. It may be restricted by a geographical area filter, to retreive only a list of stops in a particular area."
    agencies: A list of agency IDs which to retreive
        format: Format for the return value. One of ['json', 'jsonp'].
        geo_area: Geographical area filter (See the readme).
        callback: Callback function name for JSONP.
        
    """
    url = f"https://transloc-api-1-2.p.rapidapi.com/stops.{format}"
    querystring = {'agencies': agencies, }
    if geo_area:
        querystring['geo_area'] = geo_area
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transloc-api-1-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

