import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def birth_details(longitude: int, timezone: str, time: str, date: str, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Basic astrological details from the birth date and birth time"
    
    """
    url = f"https://yawin-indian-astrology.p.rapidapi.com/BirthDetails"
    querystring = {'longitude': longitude, 'timezone': timezone, 'time': time, 'date': date, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yawin-indian-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tamil_calendar_date(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the Tamil calendar date from the given English calendar date"
    
    """
    url = f"https://yawin-indian-astrology.p.rapidapi.com/TamilCalendar"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yawin-indian-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_planet_degree_to_planet_angle(degree: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a Planet Degree to  Angle Format . (Degree is 67.5666666667, converted to Angle as 67.34.0)"
    
    """
    url = f"https://yawin-indian-astrology.p.rapidapi.com/ConvertDegreeToAngle"
    querystring = {'degree': degree, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yawin-indian-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_planet_angle_to_planet_degree(angle: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a Planet Angle to  Degree Format . (Angle is 67.34.0, converted to Degree as 67.5666666667)"
    
    """
    url = f"https://yawin-indian-astrology.p.rapidapi.com/ConvertAngleToDegree"
    querystring = {'angle': angle, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yawin-indian-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sun_rise_and_sun_set_time(date: str, latitude: int, longitude: int, timezone: str='Asia/Calcutta', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the Sun Rise and Sun Set Time on a day with timezone"
    
    """
    url = f"https://yawin-indian-astrology.p.rapidapi.com/SunriseSunset"
    querystring = {'date': date, 'latitude': latitude, 'longitude': longitude, }
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yawin-indian-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_planet_position_angles(dateofbirth: str, timeofbirth: str, longitude: int=100, timezone: str='Asia/Calcutta', latitude: int=60, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the Planet position angles on a birth date and time"
    
    """
    url = f"https://yawin-indian-astrology.p.rapidapi.com/PlanetPositionAngles"
    querystring = {'dateofbirth': dateofbirth, 'timeofbirth': timeofbirth, }
    if longitude:
        querystring['longitude'] = longitude
    if timezone:
        querystring['timezone'] = timezone
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yawin-indian-astrology.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

