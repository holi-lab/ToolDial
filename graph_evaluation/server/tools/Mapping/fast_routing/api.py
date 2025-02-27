import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_route(coordinates: str, continue_straight: str=None, waypoints: str=None, alternatives: str=None, annotations: str=None, geometries: str=None, exclude: str=None, bearings: str=None, skip_waypoints: str=None, snapping: str=None, approaches: str=None, overview: str=None, radiuses: str=None, steps: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a driving route for cars (with turn-by-turn directions) by submitting (at least) a start- and endpoint."
    coordinates: At least two coordinates as start and endpoint of route, defined by longitude and latitude.
Required format: {longitude},{latitude};{longitude},{latitude}
Alternatively a polyline or polyline6 as input allowed.
        continue_straight: Forces the route to keep going straight at waypoints constraining uturns there even if it would be faster. Default is `default`, which translates to `true`.
        waypoints: Example: `{index};{index};{index}`. Treats input coordinates indicated by given indices as waypoints in returned Match object. Default is to treat all input coordinates as waypoints.
        alternatives: Show alternative routes? Either `true`, `false` (default) or a number [1,..,n] (of alternative routes to show if available).
        annotations: Either `true` , `false` (default), `nodes` , `distance` , `duration` , `datasources` , `weight` , `speed`. Defines if additional data for each coordinate should be returned.
        geometries: Either `polyline` (default), `polyline6` or `geojson`. Returned route geometry format (influences overview and per step).
        exclude: Example: `{class}`. Type of step to be avoided on route. Can be either `motorway`, `ferry` or `toll`.
        bearings: Value: `{bearing};{bearing}[;{bearing} ...]`. Limits the search to segments with given bearing in degrees towards true north in clockwise direction for each waypoint defined in coordinates.
        skip_waypoints: Default is `false`. Removes waypoints from the response. Waypoints are still calculated, but not serialized. Could be useful in case you are interested in some other part of response and do not want to transfer waste data.
        snapping: `Default` snapping avoids is_startpoint edges, `any` will snap to any edge in the graph.
        approaches: Value: `{approach};{approach}[;{approach} ...]`. Keep waypoints on curb side for each waypoint defined in coordinates path paramter.
        overview: Add overview geometry either `full` or  `simplified` according to highest zoom level it could be displayed on a map, or not at all (`false`). Default is `simplified`.
        radiuses: Values: `{radius};{radius}[;{radius} ...]`. Limits the search to given radius in meters. Set one radius for each waypoint defined in coordinates path parameter.
        steps: Get turn-by-turn direction information. Default is `false`.
        
    """
    url = f"https://fast-routing.p.rapidapi.com/route/v1/driving/{coordinates}"
    querystring = {}
    if continue_straight:
        querystring['continue_straight'] = continue_straight
    if waypoints:
        querystring['waypoints'] = waypoints
    if alternatives:
        querystring['alternatives'] = alternatives
    if annotations:
        querystring['annotations'] = annotations
    if geometries:
        querystring['geometries'] = geometries
    if exclude:
        querystring['exclude'] = exclude
    if bearings:
        querystring['bearings'] = bearings
    if skip_waypoints:
        querystring['skip_waypoints'] = skip_waypoints
    if snapping:
        querystring['snapping'] = snapping
    if approaches:
        querystring['approaches'] = approaches
    if overview:
        querystring['overview'] = overview
    if radiuses:
        querystring['radiuses'] = radiuses
    if steps:
        querystring['steps'] = steps
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-routing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

