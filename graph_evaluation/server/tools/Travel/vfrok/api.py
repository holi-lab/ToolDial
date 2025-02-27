import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def kp_index_geomagnetic_storm(q: str='EDDF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The K-index quantifies disturbances in the horizontal component of earth's magnetic field with an integer in the range 0–9 with 1 being calm and 5 or more indicating a geomagnetic storm (electronic components might like GPS might fail). It is derived from the maximum fluctuations of horizontal components observed on a magnetometer during a three-hour interval."
    
    """
    url = f"https://vfrok.p.rapidapi.com/kpindex"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vfrok.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_info(q: str='EDDF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get for a given Airport using ICAO or IATA code."
    
    """
    url = f"https://vfrok.p.rapidapi.com/airport"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vfrok.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vfr_conditions(q: str, date: str='2022-11-12T14:00:00Z', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get VFR conditions for a given Airport or Geo-Location."
    q: Provide either ICAO, IATA or geolocation. See samples for references. 
        date: Date and time to get conditions for. If not specified \\\\\\\\\\\\\\\"now\\\\\\\\\\\\\\\" will be assumed. Accepts everything that could be parsed by moments.js. 
        
    """
    url = f"https://vfrok.p.rapidapi.com/conditions"
    querystring = {'q': q, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vfrok.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

