import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ranking_at_coordinate_point(lng: str, lat: str, query: str, place_id: str, x_user_agent: str=None, zoom: str='13', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search results at the specified coordinate point and additional ranking data for the business with `place_id`."
    lng: Coordinate point longitude value.
        lat: Coordinate point latitude value.
        query: Search query / keyword.
        place_id: The Google Place ID of the business to match against in results.
        x_user_agent: Device type for the search. Default `desktop`.
        zoom: Google Maps zoom level.

**Allowed values:** 0-18.
**Default:** 13.
        
    """
    url = f"https://local-rank-tracker.p.rapidapi.com/ranking-at-coordinate"
    querystring = {'lng': lng, 'lat': lat, 'query': query, 'place_id': place_id, }
    if x_user_agent:
        querystring['X-User-Agent'] = x_user_agent
    if zoom:
        querystring['zoom'] = zoom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def full_grid_search(radius: str, lat: str, grid_size: str, place_id: str, query: str, lng: str, x_user_agent: str=None, zoom: str='13', radius_units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a full grid search and get the ranking of a specific business in every coordinate point in the grid. The grid cells / entires in the results are ordered left-to-right then top-down."
    radius: The distance between coordinate points (on the same row / column in the grid). The units of the radius are determined by the **distance_units** parameter.

**Allowed values:** 0.1-100.
**Default:** 1
        lat: Grid center coordinate point latitude value.
        grid_size: The size of the grid (i.e. 3x3, 5x5, 7x7, etc).

**Allowed values:** 3, 5, 7, 9, 11, 13, 15.
        place_id: The Google Place ID of the business to match against in results.
        query: Search query / keyword.
        lng: Grid center coordinate point longitude value.
        x_user_agent: Device type for the search. Default `desktop`.
        zoom: Google Maps zoom level to use for search on each grid point.

**Allowed values:** 0-18.
**Default:** 13.
        radius_units: Distance measurement units to use for the radius parameter (kilometers / miles).

**Allowed values:** km, mi.
**Default:** km.
        
    """
    url = f"https://local-rank-tracker.p.rapidapi.com/grid"
    querystring = {'radius': radius, 'lat': lat, 'grid_size': grid_size, 'place_id': place_id, 'query': query, 'lng': lng, }
    if x_user_agent:
        querystring['X-User-Agent'] = x_user_agent
    if zoom:
        querystring['zoom'] = zoom
    if radius_units:
        querystring['radius_units'] = radius_units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_business_locations(query: str, near: str='san jose, ca', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of found Google My Business locations based on search query, including Service Area Businesses (SAB)."
    query: Search query.
        near: Narrow down your search results by using a city, state, country or any other free-form location string (e.g. *california, usa*).
        
    """
    url = f"https://local-rank-tracker.p.rapidapi.com/places"
    querystring = {'query': query, }
    if near:
        querystring['near'] = near
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_grid_coordinate_points(radius: str, lng: str, grid_size: str, lat: str, radius_units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all grid coordinate points based on a center geocoordinate point and distance arguments."
    radius: The distance between coordinate points (on the same row / column in the grid). The units of the radius are determined by the **distance_units** parameter.

**Allowed values:** 0.1-100.
**Default:** 1
        lng: Grid center coordinate point longitude value.
        grid_size: The size of the grid (i.e. 3x3, 5x5, 7x7, etc).

**Allowed values:** 3, 5, 7, 9, 11, 13, 15.
        lat: Grid center coordinate point latitude value.
        radius_units: Distance measurement units to use for the radius parameter (kilometers / miles).

**Allowed values:** km, mi.
**Default:** km.
        
    """
    url = f"https://local-rank-tracker.p.rapidapi.com/calculate-grid-coordinates"
    querystring = {'radius': radius, 'lng': lng, 'grid_size': grid_size, 'lat': lat, }
    if radius_units:
        querystring['radius_units'] = radius_units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyword_search_at_coordinate_point(lat: str, lng: str, query: str, x_user_agent: str=None, zoom: str='13', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search results at the specified coordinate point without any rank comparison data."
    lat: Coordinate point latitude value.
        lng: Coordinate point longitude value.
        query: Search query / keyword.
        x_user_agent: Device type for the search. Default `desktop`.
        zoom: Google Maps zoom level.

**Allowed values:** 0-18.
**Default:** 13.
        
    """
    url = f"https://local-rank-tracker.p.rapidapi.com/search"
    querystring = {'lat': lat, 'lng': lng, 'query': query, }
    if x_user_agent:
        querystring['X-User-Agent'] = x_user_agent
    if zoom:
        querystring['zoom'] = zoom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

