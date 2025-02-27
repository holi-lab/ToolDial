import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_astral_data_by_city(city: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get times of various aspects of the sun for specified date and city, including dawn, sunrise, noon, sunset, dusk times. **Note**: the fuzzy search of provided city may not be the one you want due to multiple cities may have the same name, the most common one will be used. In this case, use a geocoding service to find the exact geolocation of the city, use /astral/bylocation endpoint instead."
    city: Name of the city.
        date: Local date at location specified by `lat` and `lon`, fuzzy search is supported.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/astral/bycity"
    querystring = {'city': city, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_distance_between_locations(to_lon: int, to_lat: int, from_lat: int, from_lon: int, unit: str='mi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate distance between two geolocations. We return the [Geodesic distance](https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid)  and [Great-circle distance](https://en.wikipedia.org/wiki/Great-circle_distance) for different use cases."
    to_lon: Longitude in degree.
        to_lat: Latitude in degree.
        from_lat: Latitude in degree.
        from_lon: Longitude in degree.
        unit: Length unit [mi|km]
        
    """
    url = f"https://location-and-time.p.rapidapi.com/distance/bylocation"
    querystring = {'to_lon': to_lon, 'to_lat': to_lat, 'from_lat': from_lat, 'from_lon': from_lon, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def decode_geohash(gh: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decode a [geohash](https://en.wikipedia.org/wiki/Geohash) string into a geolocation with a bounding box."
    gh: Geohash.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/geohash/decode"
    querystring = {'gh': gh, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_city_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of the city specified by name, fuzzy search is supported. Up to 10 records for cities with similar name may be returned. Information includes city name, alternative names, geolocation (latitude/longitude), population, timezone, map url, and more."
    name: Name of the city.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/city/byname"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_at_other_city(from_city: str, to_city: str, from_time: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out time at `to_city` given `from_time` and `from_city`. Daylight saving time are taken care of.  **Note**: the fuzzy search of provided city may not be the one you want due to multiple cities may have the same name, the most common one will be used. In this case, use a geocoding service to find the exact geolocation of the city, use `/timeat/bylocation` endpoint instead."
    from_city: Name of the city.
        to_city: Name of the city.
        from_time: Date and time, fuzzy search is supported.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/timeat/bycity"
    querystring = {'from_city': from_city, 'to_city': to_city, 'from_time': from_time, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_diff_by_timezone(from_timezone: str, to_timezone: str, from_time: str, to_time: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out time difference from `from_time` to `to_time` for specified timezone. Daylight saving time are taken care of."
    from_timezone: Timezone from the response of /timezone endpoint.
        to_timezone: Timezone from the response of /timezone endpoint.
        from_time: Time, fuzzy search is supported.
        to_time: Time, fuzzy search is supported.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/timediff/bytimezone"
    querystring = {'from_timezone': from_timezone, 'to_timezone': to_timezone, 'from_time': from_time, 'to_time': to_time, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_date_and_time_by_location(lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get date and time for specified location."
    lat: Latitude in degree.
        lon: Longitude in degree.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/datetime/bylocation"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_at_other_timezone(to_timezone: str, from_time: str, from_timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out time at `to_timezone` for given `from_time` and `from_timezone`. Daylight saving time are taken care of."
    to_timezone: Timezone from the response of `/timezone` endpoint.
        from_time: Date and time, fuzzy search is supported.
        from_timezone: Timezone from the response of `/timezone` endpoint.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/timeat/bytimezone"
    querystring = {'to_timezone': to_timezone, 'from_time': from_time, 'from_timezone': from_timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def encode_geohash(lon: int, lat: int, precision: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Encode a geolocation into [geohash](https://en.wikipedia.org/wiki/Geohash) string which can then used for applications such as proximity search."
    lon: Longitude in degree.
        lat: Latitude in degree.
        precision: Geohash precision.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/geohash/encode"
    querystring = {'lon': lon, 'lat': lat, }
    if precision:
        querystring['precision'] = precision
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_distance_between_cities(to_city: str, from_city: str, unit: str='mi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate distance between two cities. We return the [Geodesic distance](https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid) and [Great-circle distance](https://en.wikipedia.org/wiki/Great-circle_distance) for different use cases. Note: the fuzzy search of provided city may not be the one you want due to multiple cities may have the same name, the most common one will be used. In this case, use a geocoding service to find the exact geolocation of the city, use `/distance/bylocation` endpoint instead."
    to_city: Name of the city.
        from_city: Name of the city.
        unit: Length unit [mi|km]
        
    """
    url = f"https://location-and-time.p.rapidapi.com/distance/bycity"
    querystring = {'to_city': to_city, 'from_city': from_city, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_geolocation_by_ip(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get geographic location information about the specific IP address. Information includes the city, country, continent, geolocation with accuracy, postal code, and more."
    ip: IP address.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/geoip"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_city_by_location(lat: int, lon: int, radius: int=100000, limit: int=2, min_population: int=100000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of the city near specified geolocation (latitude/longitude). Up to 10 records for cities close to the specified location may be returned. Information includes city name, alternative names, geolocation (latitude/longitude), population, timezone, map url, and more."
    lat: Latitude in degree.
        lon: Longitude in degree.
        radius: Search radius in meters.
        limit: Max number of results.
        min_population: Minimum population.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/city/bylocation"
    querystring = {'lat': lat, 'lon': lon, }
    if radius:
        querystring['radius'] = radius
    if limit:
        querystring['limit'] = limit
    if min_population:
        querystring['min_population'] = min_population
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_country_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of the country specified by name, fuzzy search is supported. Up to 10 records for countries with similar name may be returned. Information includes country name, capital, area, population, languages, neighbors, phone code, postal code, and more."
    name: Name of the country.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/country"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_diff_by_city(from_city: str, to_city: str, to_time: str, from_time: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out time difference from `from_time` to `to_time` for specified cities. Daylight saving time are taken care of.  **Note**: the fuzzy search of provided city may not be the one you want due to multiple cities may have the same name, the most common one will be used. In this case, use a geocoding service to find the exact geolocation of the city, use `/timediff/bylocation` endpoint instead."
    from_city: Name of the city.
        to_city: Name of the city.
        to_time: Time, fuzzy search is supported.
        from_time: Time, fuzzy search is supported.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/timediff/bycity"
    querystring = {'from_city': from_city, 'to_city': to_city, 'to_time': to_time, 'from_time': from_time, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_at_other_location(to_lat: int, from_lon: int, from_time: str, to_lon: int, from_lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out time at `[to_lat, to_lon]` given `from_time` and `[from_lat, from_lon]`. Daylight saving time are taken care of."
    to_lat: Latitude in degree.
        from_lon: Longitude in degree.
        from_time: Date and time, fuzzy search is supported.
        to_lon: Longitude in degree.
        from_lat: Latitude in degree.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/timeat/bylocation"
    querystring = {'to_lat': to_lat, 'from_lon': from_lon, 'from_time': from_time, 'to_lon': to_lon, 'from_lat': from_lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supported_timezones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all supported timezones."
    
    """
    url = f"https://location-and-time.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_holidays(country: str, year: int=2023, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List holidays for the specified country and year."
    country: Name of the country, fuzzy search is supported.
        year: Year. If not provided, current year will be used.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/holiday"
    querystring = {'country': country, }
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_date_and_time_by_timezone(timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get date and time for specified timezone."
    timezone: Specify time zone from the response of `/timezone` endpoint.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/datetime/bytimezone"
    querystring = {'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_date_and_time_by_city(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current date and time for specified city, fuzzy search is supported. **Note**: the fuzzy search of provided city may not be the one you want due to multiple cities may have the same name, the most common one will be used. In this case, use a geocoding service to find the exact geolocation of the city, use `/datetime/bylocation` endpoint instead."
    city: Name of the city.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/datetime/bycity"
    querystring = {'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_astral_data_by_location(lon: int, date: str, lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get times of various aspects of the sun for specified date and location, including dawn, sunrise, noon, sunset, dusk times."
    lon: Longitude in degree.
        date: Local date at location specified by `lat` and `lon`, fuzzy search is supported.
        lat: Latitude in degree.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/astral/bylocation"
    querystring = {'lon': lon, 'date': date, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_diff_by_location(from_lon: int, from_lat: int, from_time: str, to_lon: int, to_time: str, to_lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out time difference from `from_time` to `to_time` for specified location. Daylight saving time are taken care of."
    from_lon: Longitude in degree.
        from_lat: Latitude in degree.
        from_time: Time, fuzzy search is supported.
        to_lon: Longitude in degree.
        to_time: Time, fuzzy search is supported.
        to_lat: Latitude in degree.
        
    """
    url = f"https://location-and-time.p.rapidapi.com/timediff/bylocation"
    querystring = {'from_lon': from_lon, 'from_lat': from_lat, 'from_time': from_time, 'to_lon': to_lon, 'to_time': to_time, 'to_lat': to_lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-and-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

