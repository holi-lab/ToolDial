import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_data(resample_method: str=None, model: str=None, west: int=None, south: int=None, lon: int=114, east: int=None, north: int=None, param: str='temperature', freq: str=None, end: str='2023-05-30', lat: int=23, start: str='2023-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hourly historical and forecast weather parameters in time-series format, from 1950 to 16 days ahead for any location."
    resample_method: max, mean, min, or sum. When the frequency is set to daily (D) or monthly (M), use this to specify the aggregation method.
        model: Possible values: era5, era5land, gfs, gefs, hrrr, cfs
Use to specify dataset if applicable.
        west: Latitude West. For bounding box.
        south: Latitude south. For bounding box.
        lon: Longitude(s). If location is not provided. Up to 100 locations allowed.
        east: Latitude East. For bounding box.
        north: Latitude North. For bounding box.
        freq: H (hourly), D (daily), or M (monthly). 
Defaults to H.
        end: End date. Defaults to 7 days into the future. Provided time is interpreted as UTC.
        lat: Latitude(s). If location is not provided. Up to 100 locations allowed.
        start: Start date. Defaults to 3 days into the past. Provided time is interpreted as UTC.
        
    """
    url = f"https://oikoweather.p.rapidapi.com/weather"
    querystring = {}
    if resample_method:
        querystring['resample_method'] = resample_method
    if model:
        querystring['model'] = model
    if west:
        querystring['west'] = west
    if south:
        querystring['south'] = south
    if lon:
        querystring['lon'] = lon
    if east:
        querystring['east'] = east
    if north:
        querystring['north'] = north
    if param:
        querystring['param'] = param
    if freq:
        querystring['freq'] = freq
    if end:
        querystring['end'] = end
    if lat:
        querystring['lat'] = lat
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oikoweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

