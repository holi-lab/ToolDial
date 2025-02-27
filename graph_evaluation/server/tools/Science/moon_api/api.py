import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def moon_angular_diameter(angle_units: str='am', timezone: str='+3', date_time: str='2009-07-11-09-30-00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Moon angular diameter at a given instant."
    angle_units: Sets angle units, API would use in response. Options: 
* Radians - rad
* Degrees - deg
* Turns - tr
* Arc minutes - am
* Arc seconds - as
* Milliradians - mrad 

If not specified, radians will be taken as units of length.
        timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        date_time: Parameter to set the instant when the calculation is performed. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter called timezone to set that. If parameter is not specified, the request time will be taken instead.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/angular-diameter"
    querystring = {}
    if angle_units:
        querystring['angle-units'] = angle_units
    if timezone:
        querystring['timezone'] = timezone
    if date_time:
        querystring['date-time'] = date_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_illumination(timezone: str='+3', date_time: str='2009-07-11-09-30-00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns  fraction of the Moon which is lit at some instant in time."
    timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        date_time: Parameter to set the instant when the calculation is performed. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter called timezone to set that. If parameter is not specified, the request time will be taken instead.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/illumination"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if date_time:
        querystring['date-time'] = date_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_horizontal_position_position_on_the_sky(location: str, angle_units: str='deg', timezone: str='+3', date_time: str='2009-07-11-09-30-00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Moon horizontal position (azimuth and altitude). Azimuth and altitude are expressed as angles."
    location: Location on Earth surface from which calculations are performed. It's format is LAT, LON. Latitude and longitude must be specified in degrees. Latitude is positive in northern hemisphere and negative in southern. Longitude is positive east of the 0 meridian and negative west. 
        angle_units: Sets angle units, API would use in response. Options: 
* Radians - rad
* Degrees - deg
* Turns - tr
* Arc minutes - am
* Arc seconds - as
* Milliradians - mrad 

If not specified, radians will be taken as units of length.
        timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        date_time: Parameter to set the instant when the calculation is performed. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter called timezone to set that. If parameter is not specified, the request time will be taken instead.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/horizontal-position"
    querystring = {'location': location, }
    if angle_units:
        querystring['angle-units'] = angle_units
    if timezone:
        querystring['timezone'] = timezone
    if date_time:
        querystring['date-time'] = date_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_celestial_position(timezone: str='+3', date_time: str='2009-07-11-09-30-00', angle_units: str='deg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Moon celestial position (right ascension and declination) at some instant. Right ascension and declination are expressed as angles."
    timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        date_time: Parameter to set the instant when the calculation is performed. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter called timezone to set that. If parameter is not specified, the request time will be taken instead.
        angle_units: Sets angle units, API would use in response. Options: 
* Radians - rad
* Degrees - deg
* Turns - tr
* Arc minutes - am
* Arc seconds - as
* Milliradians - mrad 

If not specified, radians will be taken as units of length.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/celestial-position"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if date_time:
        querystring['date-time'] = date_time
    if angle_units:
        querystring['angle-units'] = angle_units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_eliptic_position(timezone: str='+3', angle_units: str='deg', date_time: str='2009-07-11-09-30-00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Moon eliptic position (latitude and longitude) at some instant. Latitude and longitude are expressed as angles."
    timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        angle_units: Sets angle units, API would use in response. Options: 
* Radians - rad
* Degrees - deg
* Turns - tr
* Arc minutes - am
* Arc seconds - as
* Milliradians - mrad 

If not specified, radians will be taken as units of length.
        date_time: Parameter to set the instant when the calculation is performed. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter called timezone to set that. If parameter is not specified, the request time will be taken instead.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/eliptic-position"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if angle_units:
        querystring['angle-units'] = angle_units
    if date_time:
        querystring['date-time'] = date_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_hour_angle(location: str, timezone: str='+3', date_time: str='2009-07-11-09-30-00', angle_units: str='deg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns moon hour angle at some instant."
    location: Location on Earth surface from which calculations are performed. It's format is LAT, LON. Latitude and longitude must be specified in degrees. Latitude is positive in northern hemisphere and negative in southern. Longitude is positive east of the 0 meridian and negative west. 
        timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        date_time: Parameter to set the instant when the calculation is performed. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter called timezone to set that. If parameter is not specified, the request time will be taken instead.
        angle_units: Sets angle units, API would use in response. Options:
* Radians - rad
* Degrees - deg
* Turns - tr
* Arc minutes - am
* Arc seconds - as
* Milliradians - mrad

If not specified, radians will be taken as units of length.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/hour-angle"
    querystring = {'location': location, }
    if timezone:
        querystring['timezone'] = timezone
    if date_time:
        querystring['date-time'] = date_time
    if angle_units:
        querystring['angle-units'] = angle_units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_phase(date_time: str='2009-07-11-09-30-00', angle_units: str='deg', timezone: str='+3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Moon phase and other data connected to it at some instant."
    date_time: Parameter to set the point in time when the calculation is done. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter to set that. If parameter is not specified, the request time will be taken instead.
        angle_units: Sets angle units, API would use in response. Options:
* Radians - rad
* Degrees - deg
* Turns - tr
* Arc minutes - am
* Arc seconds - as
* Milliradians - mrad

If not specified, radians will be taken as units of length.
        timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/phase"
    querystring = {}
    if date_time:
        querystring['date-time'] = date_time
    if angle_units:
        querystring['angle-units'] = angle_units
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_general_data(length_units: str='km', angle_units: str='deg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns general data about the Moon."
    length_units: Sets length units, API would use in response. Options: 
* Meters - m
* Kilometers - km
* Miles - mi
* Nautical miles - nmi
* Astronomical units - au
* Light seconds - ls
* Yards - yd
* Foots - ft
* Inches - in
* Centimeters - cm

If not specified, meters will be taken as units of length.
        angle_units: Sets angle units, API would use in response. Options: 
* Radians - rad
* Degrees - deg
* Turns - tr
* Arc minutes - am
* Arc seconds - as
* Milliradians - mrad 

If not specified, radians will be taken as units of length.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/general-data"
    querystring = {}
    if length_units:
        querystring['length-units'] = length_units
    if angle_units:
        querystring['angle-units'] = angle_units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moon_distance(timezone: str='+3', date_time: str='2009-07-11-09-30-00', length_units: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Moon distance at some instant."
    timezone: Sets timezone for the time. If not specified, UTC+0 timezone would be taken instead.
        date_time: Parameter to set the instant when the calculation is performed. It's format is YYYY-MM-DD-YY-MM-SS. If the time is not in UTC timezone, you have to use special parameter called timezone to set that. If parameter is not specified, the request time will be taken instead.
        length_units: Sets length units, API would use in response. Options: 
* Meters - m
* Kilometers - km
* Miles - mi
* Nautical miles - nmi
* Astronomical units - au
* Light seconds - ls
* Yards - yd
* Foots - ft
* Inches - in
* Centimeters - cm

If not specified, meters will be taken as units of length.
        
    """
    url = f"https://moon-api1.p.rapidapi.com/distance"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if date_time:
        querystring['date-time'] = date_time
    if length_units:
        querystring['length-units'] = length_units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moon-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

