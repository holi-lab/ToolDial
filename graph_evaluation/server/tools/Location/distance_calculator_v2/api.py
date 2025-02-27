import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def one_to_one(start_point: str, end_point: str, decimal_places: int=2, unit: str='miles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance between two points"
    start_point: Coordinates of  start point (latitude, longitude)
        end_point: Coordinates of end point (latitude, longitude)
        decimal_places: Default: 16 . The value can be from 0 to 16
        unit: Default: kilometers. May have values: kilometers, meters, miles, feet, nautical_miles
        
    """
    url = f"https://distance-calculator.p.rapidapi.com/v1/one_to_one"
    querystring = {'start_point': start_point, 'end_point': end_point, }
    if decimal_places:
        querystring['decimal_places'] = decimal_places
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distance-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def one_to_many(start_point: str, decimal_places: int=2, end_point_8: str='(61.402022,23.717415)', end_point_1: str='(61.280642,9.696496)', end_point_7: str='(41.192840,-2.550522)', end_point_3: str='(63.946372,-17.301934)', end_point_4: str='(29.783423,-82.937419)', end_point_2: str='(42.335321,-71.023516)', end_point_9: str='(57.079849,-116.604973)', end_point_6: str='(40.116758,-111.149673)', end_point_5: str='(39.177734,-123.404589)', unit: str='miles', end_point_10: str='(39.208789,-8.240891)', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance between start point and up to 10 end points"
    start_point: Coordinates of start point (latitude, longitude)
        decimal_places: Default: 16 . The value can be from 0 to 16
        end_point_8: Coordinates of end point (latitude, longitude)
        end_point_1: Coordinates of end point (latitude, longitude)
        end_point_7: Coordinates of end point (latitude, longitude)
        end_point_3: Coordinates of end point (latitude, longitude)
        end_point_4: Coordinates of end point (latitude, longitude)
        end_point_2: Coordinates of end point (latitude, longitude)
        end_point_9: Coordinates of end point (latitude, longitude)
        end_point_6: Coordinates of end point (latitude, longitude)
        end_point_5: Coordinates of end point (latitude, longitude)
        unit: Default: kilometers. May have values: kilometers, meters, miles, feet, nautical_miles
        end_point_10: Coordinates of end point (latitude, longitude)
        
    """
    url = f"https://distance-calculator.p.rapidapi.com/v1/one_to_many"
    querystring = {'start_point': start_point, }
    if decimal_places:
        querystring['decimal_places'] = decimal_places
    if end_point_8:
        querystring['end_point_8'] = end_point_8
    if end_point_1:
        querystring['end_point_1'] = end_point_1
    if end_point_7:
        querystring['end_point_7'] = end_point_7
    if end_point_3:
        querystring['end_point_3'] = end_point_3
    if end_point_4:
        querystring['end_point_4'] = end_point_4
    if end_point_2:
        querystring['end_point_2'] = end_point_2
    if end_point_9:
        querystring['end_point_9'] = end_point_9
    if end_point_6:
        querystring['end_point_6'] = end_point_6
    if end_point_5:
        querystring['end_point_5'] = end_point_5
    if unit:
        querystring['unit'] = unit
    if end_point_10:
        querystring['end_point_10'] = end_point_10
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distance-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple(long_2: str, long_1: str, lat_1: str, lat_2: str, unit: str='miles', decimal_places: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance between two points"
    long_2: Longitude of Point 2
        long_1: Longitude of Point 1
        lat_1: Latitude of Point 1
        lat_2: Latitude of Point 2
        unit: Default: kilometers . May have values: kilometers, meters, miles, feet, nautical_miles
        decimal_places: Default: 16 . The value can be from 0 to 16
        
    """
    url = f"https://distance-calculator.p.rapidapi.com/distance/simple"
    querystring = {'long_2': long_2, 'long_1': long_1, 'lat_1': lat_1, 'lat_2': lat_2, }
    if unit:
        querystring['unit'] = unit
    if decimal_places:
        querystring['decimal_places'] = decimal_places
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distance-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

