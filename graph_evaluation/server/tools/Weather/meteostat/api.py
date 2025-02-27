import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def monthly_point_data(lon: int, end: str, start: str, lat: int, freq: str=None, units: str=None, alt: int=43, model: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides historical monthly statistics for a geographic location. The data provided through this endpoint is aggregated from multiple governmental interfaces."
    lon: The point's longitude.
        end: The end date of the period (YYYY-MM-DD).
        start: The start date of the period (YYYY-MM-DD).
        lat: The point's latitude.
        freq: The time frequency of the records. Can be used for custom aggregation. Default is *null*.
        units: The unit system of the meteorological parameters. Default is metric.
        alt: The point's elevation.
        model: A boolean parameter which specifies whether missing data should be substituted with statistically optimized model data. Default is *true*.
        
    """
    url = f"https://meteostat.p.rapidapi.com/point/monthly"
    querystring = {'lon': lon, 'end': end, 'start': start, 'lat': lat, }
    if freq:
        querystring['freq'] = freq
    if units:
        querystring['units'] = units
    if alt:
        querystring['alt'] = alt
    if model:
        querystring['model'] = model
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def monthly_station_data(station: str, end: str, start: str, units: str=None, model: bool=None, freq: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides historical monthly statistics for a particular weather station. The data provided through this endpoint is aggregated from multiple governmental interfaces."
    station: The Meteostat weather station identifier.
        end: The end date of the period (YYYY-MM-DD).
        start: The start date of the period (YYYY-MM-DD).
        units: The unit system of the meteorological parameters. Default is metric.
        model: A boolean parameter which specifies whether missing data should be substituted with statistically optimized model data. Default is *true*.
        freq: The time frequency of the records. Can be used for custom aggregation. Default is *null*.
        
    """
    url = f"https://meteostat.p.rapidapi.com/stations/monthly"
    querystring = {'station': station, 'end': end, 'start': start, }
    if units:
        querystring['units'] = units
    if model:
        querystring['model'] = model
    if freq:
        querystring['freq'] = freq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def station_meta_data(wmo: str=None, icao: str=None, is_id: str='10637', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides meta data for a particular weather station."
    wmo: The WMO identifier of a weather station.
        icao: The ICAO identifier of a weather station.
        is_id: The Meteostat identifier of a weather station.
        
    """
    url = f"https://meteostat.p.rapidapi.com/stations/meta"
    querystring = {}
    if wmo:
        querystring['wmo'] = wmo
    if icao:
        querystring['icao'] = icao
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_point_data(lon: int, lat: int, start: str, end: str, units: str=None, freq: str=None, model: bool=None, alt: int=184, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides historical daily statistics for a geographic location. The data provided through this endpoint is aggregated from multiple governmental interfaces."
    lon: The point's longitude.
        lat: The point's latitude.
        start: The start date of the period (YYYY-MM-DD).
        end: The end date of the period (YYYY-MM-DD).
        units: The unit system of the meteorological parameters. Default is metric.
        freq: The time frequency of the records. Can be used for custom aggregation. Default is *null*.
        model: A boolean parameter which specifies whether missing data should be substituted with statistically optimized model data. Default is *true*.
        alt: The point's elevation.
        
    """
    url = f"https://meteostat.p.rapidapi.com/point/daily"
    querystring = {'lon': lon, 'lat': lat, 'start': start, 'end': end, }
    if units:
        querystring['units'] = units
    if freq:
        querystring['freq'] = freq
    if model:
        querystring['model'] = model
    if alt:
        querystring['alt'] = alt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly_point_data(lon: int, lat: int, end: str, start: str, tz: str='America/Toronto', alt: int=113, freq: str=None, model: bool=None, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides historical hourly observations for a geographic location. The data provided through this endpoint is aggregated from multiple governmental interfaces."
    lon: The point's longitude.
        lat: The point's latitude.
        end: The end date of the period (YYYY-MM-DD).
        start: The start date of the period (YYYY-MM-DD).
        tz: The time zone according to the tz database. Default is UTC.
        alt: The point's elevation.
        freq: The time frequency of the records. Can be used for custom aggregation. Default is *null*.
        model: A boolean parameter which specifies whether missing data should be substituted with statistically optimized model data. Default is *true*.
        units: The unit system of the meteorological parameters. Default is metric.
        
    """
    url = f"https://meteostat.p.rapidapi.com/point/hourly"
    querystring = {'lon': lon, 'lat': lat, 'end': end, 'start': start, }
    if tz:
        querystring['tz'] = tz
    if alt:
        querystring['alt'] = alt
    if freq:
        querystring['freq'] = freq
    if model:
        querystring['model'] = model
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def station_climate_data(station: str, end: int=1990, start: int=1961, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides climate normals for a particular weather station."
    station: The Meteostat weather station identifier.
        end: The end year of the reference period.
        start: The start year of the reference period.
        units: The unit system of the meteorological parameters. Default is metric.
        
    """
    url = f"https://meteostat.p.rapidapi.com/stations/normals"
    querystring = {'station': station, }
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def point_climate_data(lat: int, lon: int, units: str=None, end: int=1990, start: int=1961, alt: int=26, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides climate normals for any geo location."
    lat: The point's latitude.
        lon: The point's longitude.
        units: The unit system of the meteorological parameters. Default is metric.
        end: The end year of the reference period.
        start: The start year of the reference period.
        alt: The point's elevation.
        
    """
    url = f"https://meteostat.p.rapidapi.com/point/normals"
    querystring = {'lat': lat, 'lon': lon, }
    if units:
        querystring['units'] = units
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    if alt:
        querystring['alt'] = alt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby_stations(lon: int, lat: int, limit: int=None, radius: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a list of nearby weather stations for a given geographical location."
    lon: The location's longitude.
        lat: The location's latitude.
        limit: The maximum number of weather stations. Default is 10.
        radius: The meter radius to search in. Default is 100000.
        
    """
    url = f"https://meteostat.p.rapidapi.com/stations/nearby"
    querystring = {'lon': lon, 'lat': lat, }
    if limit:
        querystring['limit'] = limit
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly_station_data(start: str, station: str, end: str, model: bool=None, tz: str='Europe/Berlin', units: str=None, freq: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides historical hourly observations for a particular weather station. The data provided through this endpoint is aggregated from multiple governmental interfaces."
    start: The start date of the period (YYYY-MM-DD).
        station: The Meteostat weather station identifier.
        end: The end date of the period (YYYY-MM-DD).
        model: A boolean parameter which specifies whether missing observations should be substituted with statistically optimized model data. Default is *true*.
        tz: The time zone according to the tz database. Default is UTC.
        units: The unit system of the meteorological parameters. Default is metric.
        freq: The time frequency of the records. Can be used for custom aggregation. Default is *null*.
        
    """
    url = f"https://meteostat.p.rapidapi.com/stations/hourly"
    querystring = {'start': start, 'station': station, 'end': end, }
    if model:
        querystring['model'] = model
    if tz:
        querystring['tz'] = tz
    if units:
        querystring['units'] = units
    if freq:
        querystring['freq'] = freq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_station_data(start: str, station: str, end: str, units: str=None, model: bool=None, freq: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides historical daily statistics for a particular weather station. The data provided through this endpoint is aggregated from multiple governmental interfaces."
    start: The start date of the period (YYYY-MM-DD).
        station: The Meteostat weather station identifier.
        end: The end date of the period (YYYY-MM-DD).
        units: The unit system of the meteorological parameters. Default is metric.
        model: A boolean parameter which specifies whether missing data should be substituted with statistically optimized model data. Default is *true*.
        freq: The time frequency of the records. Can be used for custom aggregation. Default is *null*.
        
    """
    url = f"https://meteostat.p.rapidapi.com/stations/daily"
    querystring = {'start': start, 'station': station, 'end': end, }
    if units:
        querystring['units'] = units
    if model:
        querystring['model'] = model
    if freq:
        querystring['freq'] = freq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

