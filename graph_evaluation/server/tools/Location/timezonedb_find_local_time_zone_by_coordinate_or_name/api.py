import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def main(zone: str='America/Los_Angeles', lat: int=34, lng: int=-118, time: int=1358474681, format: str='xml', callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return country code, GMT offset, daylight saving time, time zone name, and current timestamp."
    zone: Time zone name . You can refer the name from http://timezonedb.com/time-zones. Required if lat & lng fields are not provided.
        lat: Latitude of a city. This field is required when "zone" is not specified.
        lng: Longitude of a city. This field is required when "zone" is not specified.
        time: Unix timestamp to be converted into local time. For more information of Unix time, please refer to http://en.wikipedia.org/wiki/Unix_time
        format: The forat of the returned result. Default is XML formatted.
        callback: Use for JavaScript JSON callback.
        
    """
    url = f"https://timezonedb.p.rapidapi.com/"
    querystring = {}
    if zone:
        querystring['zone'] = zone
    if lat:
        querystring['lat'] = lat
    if lng:
        querystring['lng'] = lng
    if time:
        querystring['time'] = time
    if format:
        querystring['format'] = format
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timezonedb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

