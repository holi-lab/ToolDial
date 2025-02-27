import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def raw_data(datetime_end: str, lat: str, datetime_ini: str, lon: str, distance: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Official crime reports for 30+ cities across the US"
    datetime_end: End date/time (format yyyy-MM-dd'T'HH: mm: ss.SSS'Z)
        lat: latitude, format +/-XX.YYYY
        datetime_ini: Initial date/time (format yyyy-MM-dd'T'HH: mm: ss.SSS'Z)
        lon: longitude, format +/-XXX.YYYY
        distance: radio around lat and long coordinates to filter reports, in Miles, Yards, Feets, Kilometers or Meters. Example: 10mi, 10yd, 10ft, 10km, 10m respectively.
        page: (default '1'), if the user set this parameter the API will return the next 10 incidents ordered by incident_date desc (if any). For example, if set to '2', it will return reports 11 to 20 (if any).
        
    """
    url = f"https://crimeometer.p.rapidapi.comraw-data"
    querystring = {'datetime_end': datetime_end, 'lat': lat, 'datetime_ini': datetime_ini, 'lon': lon, 'distance': distance, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crimeometer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stats(lon: str, datetime_ini: str, datetime_end: str, lat: str, distance: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The access point stats is a new access point introduced in the version 1, currently only provides information about the total number of incidents, and the total number of incidents per type in a defined area and time period."
    lon: longitude, format +/-XXX.YYYY
        datetime_ini: Initial date/time (format yyyy-MM-dd'T'HH: mm: ss.SSS'Z)
        datetime_end: End date/time (format yyyy-MM-dd'T'HH: mm: ss.SSS'Z)
        lat: latitude, format +/-XX.YYYY
        distance: radio around lat and long coordinates to filter reports, in Miles, Yards, Feets, Kilometers or Meters. Example: 10mi, 10yd, 10ft, 10km, 10m respectively.
        
    """
    url = f"https://crimeometer.p.rapidapi.comstats"
    querystring = {'lon': lon, 'datetime_ini': datetime_ini, 'datetime_end': datetime_end, 'lat': lat, }
    if distance:
        querystring['distance'] = distance
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crimeometer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

