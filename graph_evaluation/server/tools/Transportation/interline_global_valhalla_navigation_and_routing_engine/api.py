import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tracerouteget(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The trace_route action takes the costing mode and a list of latitude,longitude coordinates, for example, from a GPS trace, to turn them into a route with the shape snapped to the road network and a set of guidance directions. You might use this to take a GPS trace from a bike route into a set of narrative instructions so you can re-create your trip or share it with others. See https://valhalla.readthedocs.io/en/latest/map-matching/api-reference.md#example-map-matching-requests"
    json: JSON payload for the routing engine request. See documentation for the possible contents.
        
    """
    url = f"https://interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com/trace_route"
    querystring = {'json': json, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def routeget(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Valhalla's routing service (a.k.a. turn-by-turn), is an open-source routing service that lets you integrate routing and navigation into a web or mobile application. See https://valhalla.readthedocs.io/en/latest/turn-by-turn/api-reference.md"
    json: JSON payload for the routing engine request. See documentation for the possible contents.
        
    """
    url = f"https://interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com/route"
    querystring = {'json': json, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def optimized_route(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Optimized Route service provides a quick computation of time and distance between a set of location sources and location targets and returns them in an optimized route order, along with the shape. See https://valhalla.readthedocs.io/en/latest/optimized/api-reference.md"
    json: JSON payload for the routing engine request. See documentation for the possible contents.
        
    """
    url = f"https://interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com/optimized_route"
    querystring = {'json': json, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traceattributesget(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The trace_attributes action takes the costing mode and a GPS trace or latitude,longitude positions and returns detailed attribution along the portion of the route. This includes details for each section of road along the path, as well as any intersections along the path. See https://valhalla.readthedocs.io/en/latest/api/map-matching/api-reference/#trace-attributes-action"
    json: JSON payload for the routing engine request. See documentation for the possible contents.
        
    """
    url = f"https://interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com/trace_attributes"
    querystring = {'json': json, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matrixget(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Valhalla's time-distance matrix service provides a quick computation of time and distance between a set of locations and returns them to you in the resulting matrix table. See https://valhalla.readthedocs.io/en/latest/matrix/api-reference.md"
    json: JSON payload for the routing engine request. See documentation for the possible contents.
        
    """
    url = f"https://interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com/sources_to_targets"
    querystring = {'json': json, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locate(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Valhalla's locate service, is an open-source service that provides detailed information about streets and intersections close to an input point with some added matching criteria. This allows for tight integration in routing and navigation applications on web or mobile. See https://valhalla.readthedocs.io/en/latest/locate/api-reference.md"
    json: JSON payload for the routing engine request. See documentation for the possible contents.
        
    """
    url = f"https://interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com/locate"
    querystring = {'json': json, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isochrone(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An isochrone is a line that connects points of equal travel time about a given location, from the Greek roots of iso for equal and chrone for time. Valhalla's isochrone service computes areas that are reachable within specified time intervals from a location, and returns the reachable regions as contours of polygons or lines that you can display on a map. See https://valhalla.readthedocs.io/en/latest/isochrone/api-reference.md"
    json: JSON payload for the routing engine request. See documentation for the possible contents.
        
    """
    url = f"https://interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com/isochrone"
    querystring = {'json': json, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interline-global-valhalla-navigation-and-routing-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

