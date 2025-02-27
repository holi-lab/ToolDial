import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def forecast(latitude: str, longitude: str, exclude: str=None, units: str='auto', extend: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A Forecast Request returns the current weather conditions, a minute-by-minute forecast for the next hour (where available), an hour-by-hour forecast for the next 48 hours, and a day-by-day forecast for the next week."
    latitude: The latitude of a location (in decimal degrees). Positive is north, negative is south.
        longitude: The longitude of a location (in decimal degrees). Positive is east, negative is west.
        exclude: Exclude some number of data blocks from the API response. This is useful for reducing latency and saving cache space. The value blocks should be a comma-delimeted list (without spaces) of any of the following:  currently, minutely, hourly, daily, alerts, flags
        units: Return weather conditions in the requested units. [units] should be one of the following:  auto: automatically select units based on geographic location ca: same as si, except that windSpeed and windGust are in kilometers per hour uk2: same as si, except that nearestStormDistance and visibility are in miles, and windSpeed and windGust in miles per hour us: Imperial units (the default) si: SI units
        extend: When extend=hourly, return hour-by-hour data for the next 168 hours, instead of the next 48. When using this option, we strongly recommend enabling HTTP compression.
        lang: Return summary properties in the desired language. (Note that units in the summary will be set according to the units parameter, so be sure to set both parameters appropriately.) language should be 2-letter language code. English is default
        
    """
    url = f"https://dark-sky.p.rapidapi.com/{latitude},{longitude}"
    querystring = {}
    if exclude:
        querystring['exclude'] = exclude
    if units:
        querystring['units'] = units
    if extend:
        querystring['extend'] = extend
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dark-sky.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_machine(time: str, longitude: str, latitude: str, lang: str=None, exclude: str=None, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A Time Machine Request returns the observed (in the past) or forecasted (in the future) hour-by-hour weather and daily weather conditions for a particular date."
    time: Either be a UNIX time (that is, seconds since midnight GMT on 1 Jan 1970) or a string formatted as follows: [YYYY]-[MM]-[DD]T[HH]:[MM]:[SS][timezone]. timezone should either be omitted (to refer to local time for the location being requested), Z (referring to GMT time), or +[HH][MM] or -[HH][MM] for an offset from GMT in hours and minutes. The timezone is only used for determining the time of the request; the response will always be relative to the local time zone.
        longitude: The longitude of a location (in decimal degrees). Positive is east, negative is west.
        latitude: The latitude of a location (in decimal degrees). Positive is north, negative is south.
        lang: Return summary properties in the desired language. (Note that units in the summary will be set according to the units parameter, so be sure to set both parameters appropriately.) Use 2-letter language code (full list in API Details). en is default.
        exclude: Exclude some number of data blocks from the API response. This is useful for reducing latency and saving cache space. The value blocks should be a comma-delimeted list (without spaces) of any of the following:  currently, minutely, hourly, daily, alerts, flags
        units: Return weather conditions in the requested units. [units] should be one of the following:  auto: automatically select units based on geographic location ca: same as si, except that windSpeed is in kilometers per hour uk2: same as si, except that nearestStormDistance and visibility are in miles and windSpeed is in miles per hour us: Imperial units (the default) si: SI units
        
    """
    url = f"https://dark-sky.p.rapidapi.com/{latitude},{longitude},{time}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if exclude:
        querystring['exclude'] = exclude
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dark-sky.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

