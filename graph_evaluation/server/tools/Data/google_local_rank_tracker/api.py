import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keyword_search_at_coordinates(query: str, lng: int, lat: int, zoom: int=13, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a Keyword search at a Coordinate (lat/lng) for all the business."
    query: The **Google Place ID** of the business or the **Business Name** to match against in results. Use the **match_type** parameter to specify.
        lng: Grid **center** coordinate point **longitude** value.
        lat: Grid **center** coordinate point **latitude** value.
        zoom: Google Maps **zoom level **to use for search on each grid point.

**Allowed values**: 0-18.
**Default**: 13.
        
    """
    url = f"https://google-local-rank-tracker.p.rapidapi.com/geopoint-search"
    querystring = {'query': query, 'lng': lng, 'lat': lat, }
    if zoom:
        querystring['zoom'] = zoom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_geogrid_coordinate_points(lng: int, lat: int, width: int=5, distance_unit: str='km', distance: int=1, grid_size: int=5, height: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all grid coordinate points based on a center geocoordinate point and distance arguments."
    lng: Grid **center** coordinate point **longitude** value.
        lat: Grid **center** coordinate point **latitude** value.
        width: The **width** of the grid in location points for non-square grid searches.

**Allowed values**: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25.
**Default**: 9
        distance_unit: Distance measurement units to use for the distance parameter (kilometers / miles).

**Allowed values**: km, mi.
**Default**: km.
        distance: The distance between coordinate points (on the same row / column in the grid).

The units of the radius are determined by the distance_units parameter.

**Allowed values**: 0.1-100.
**Default**: 1
        grid_size: The **size** of the grid (i.e. 3x3, 5x5, 7x7, etc).

**Allowed values**: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25.
**Default**: 9
        height: The **height** of the grid in location points for non-square grid searches.

**Allowed values**: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25.
**Default**: 9
        
    """
    url = f"https://google-local-rank-tracker.p.rapidapi.com/geogrid-coordinates"
    querystring = {'lng': lng, 'lat': lat, }
    if width:
        querystring['width'] = width
    if distance_unit:
        querystring['distance_unit'] = distance_unit
    if distance:
        querystring['distance'] = distance
    if grid_size:
        querystring['grid_size'] = grid_size
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyword_search_with_ranking(query: str, lng: int, match_value: str, lat: int, zoom: int=13, match_type: str='place_id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a Keyword search at a Coordinate (lat/lng) for all the business and get the ranking of a specific business at that point. The ranking data is determined by the business place_id or business name."
    query: The **Google Place ID** of the business or the **Business Name** to match in results. Use the **match_type** parameter to specify choice.
        lng: Grid **center** coordinate point **longitude** value.
        match_value: Search query / **keyword**.
        lat: Grid **center** coordinate point **latitude** value.
        zoom: Google Maps **zoom level **to use for search on each grid point.

**Allowed values**: 0-18.
**Default**: 13.
        match_type: The type of match to perform for ranking.

**Allowed values**: place_id, name.
**Default**: place_id
        
    """
    url = f"https://google-local-rank-tracker.p.rapidapi.com/geopoint"
    querystring = {'query': query, 'lng': lng, 'match_value': match_value, 'lat': lat, }
    if zoom:
        querystring['zoom'] = zoom
    if match_type:
        querystring['match_type'] = match_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geogrid_seach_with_ranking(match_value: str, query: str, lng: int, lat: int, zoom: int=13, match_type: str='place_id', distance_unit: str='km', width: int=5, height: int=5, grid_size: int=5, distance: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a full grid search and get the ranking of a specific business in every coordinate point in the grid. The grid cells / entires in the results are ordered left-to-right then top-to-bottom.  Contains additional ranking data for the business by place_id or business name."
    match_value: Search query / **keyword**.
        query: The **Google Place ID** of the business or the **Business Name** to match in results. Use the **match_type** parameter to specify choice.
        lng: Grid **center** coordinate point **longitude** value.
        lat: Grid **center** coordinate point **latitude** value.
        zoom: Google Maps **zoom level **to use for search on each grid point.

**Allowed values**: 0-18.
**Default**: 13.
        match_type: The type of match to perform for ranking.

**Allowed values**: place_id, name.
**Default**: place_id
        width: The **width** of the grid in location points for non-square grid searches.

**Allowed values**: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25.
**Default**: 9
        height: The **height** of the grid in location points for non-square grid searches.

**Allowed values**: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25.
**Default**: 9
        grid_size: The size of the grid (i.e. 3x3, 5x5, 7x7, etc).

**Allowed values**: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25
**Default**: 9
        distance: The distance between coordinate points (on the same row / column in the grid). The units of the radius are determined by the distance_units parameter.

**Allowed values**: 0.1-100.
**Default**: 1
        
    """
    url = f"https://google-local-rank-tracker.p.rapidapi.com/geogrid"
    querystring = {'match_value': match_value, 'query': query, 'lng': lng, 'lat': lat, }
    if zoom:
        querystring['zoom'] = zoom
    if match_type:
        querystring['match_type'] = match_type
    if distance_unit:
        querystring['distance_unit'] = distance_unit
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if grid_size:
        querystring['grid_size'] = grid_size
    if distance:
        querystring['distance'] = distance
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-local-rank-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

