import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def simple_radiation_forecast(latitude: int, longitude: int, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The simple radiation request returns detailed solar radiation data for the next week based only on your latitude and longitude."
    latitude: Latitude
        longitude: Longitude
        format: Response format: json, csv, xml
        
    """
    url = f"https://solcast.p.rapidapi.com/radiation/forecasts"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "solcast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_pv_power_forecast(capacity: int, latitude: int, longitude: int, azimuth: int=None, install_date: str=None, loss_factor: str=None, tilt: int=23, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The simple PV power request returns a first-guess PV power output forecast, based on your specified latitude and longitude plus some basic PV system characteristics."
    capacity: The capacity of the system, in Watts.
        latitude: Latitude
        longitude: Longitude
        azimuth: The angle (degrees) from true north that the PV system is facing, if tilted.  Must be between -180 and 180.  An azimuth of 0 means the system is facing true north.
        install_date: The date (YYYYMMMDD) of installation of the PV system.  We use this to estimate your loss factor based on ageing of your system.  If you provide a loss_factor this date will be ignored.
        loss_factor: A factor by which to reduce your output forecast fom the full capacity based on characteristics of the PV array or inverter.
        tilt: The angle (degrees) that the PV system is tilted off the horizontal.  Must be between 0 and 90. A tilt of 0 means that it is facing directly upwards, and 90 means the system is vertical and facing the horizon.
        format: Response format: json, csv, xml
        
    """
    url = f"https://solcast.p.rapidapi.com/pv_power/forecasts"
    querystring = {'capacity': capacity, 'latitude': latitude, 'longitude': longitude, }
    if azimuth:
        querystring['azimuth'] = azimuth
    if install_date:
        querystring['install_date'] = install_date
    if loss_factor:
        querystring['loss_factor'] = loss_factor
    if tilt:
        querystring['tilt'] = tilt
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "solcast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

