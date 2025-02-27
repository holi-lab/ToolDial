import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_planets(max_period: int=None, max_temperature: int=None, offset: int=None, max_distance_light_year: int=None, min_distance_light_year: int=None, max_mass: int=None, max_semi_major_axis: int=None, min_mass: int=None, min_semi_major_axis: int=None, name: str='Mars', min_temperature: int=None, max_radius: int=None, min_radius: int=None, min_period: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of planets matching specified parameters. Returns at most 30 results. To access more than 30 results, use the offset parameter to offset results in multiple API calls."
    max_period: maximum orbital period of the planet in Earth days.
        max_temperature: maximum average surface temperature of the planet in Kelvin.
        offset: number of results to offset for pagination.
        max_distance_light_year: maximum distance the planet is from Earth in light years.
        min_distance_light_year: minimum distance the planet is from Earth in light years.
        max_mass: maximum mass of the planet in Jupiters (1 Jupiter = 1.898 × 10^27 kg).
        max_semi_major_axis: maximum semi major axis of planet in astronomical units (AU).
        min_mass: minimum mass of the planet in Jupiters (1 Jupiter = 1.898 × 10^27 kg).
        min_semi_major_axis: minimum semi major axis of planet in astronomical units (AU).
        name:  name of the planet.
        min_temperature: minimum average surface temperature of the planet in Kelvin.
        max_radius: maximum average radius of the planet in Jupiters (1 Jupiter = 69911 km).
        min_radius: minimum average radius of the planet in Jupiters (1 Jupiter = 69911 km).
        min_period: minimum orbital period of the planet in Earth days.
        
    """
    url = f"https://planets-by-api-ninjas.p.rapidapi.com/v1/planets"
    querystring = {}
    if max_period:
        querystring['max_period'] = max_period
    if max_temperature:
        querystring['max_temperature'] = max_temperature
    if offset:
        querystring['offset'] = offset
    if max_distance_light_year:
        querystring['max_distance_light_year'] = max_distance_light_year
    if min_distance_light_year:
        querystring['min_distance_light_year'] = min_distance_light_year
    if max_mass:
        querystring['max_mass'] = max_mass
    if max_semi_major_axis:
        querystring['max_semi_major_axis'] = max_semi_major_axis
    if min_mass:
        querystring['min_mass'] = min_mass
    if min_semi_major_axis:
        querystring['min_semi_major_axis'] = min_semi_major_axis
    if name:
        querystring['name'] = name
    if min_temperature:
        querystring['min_temperature'] = min_temperature
    if max_radius:
        querystring['max_radius'] = max_radius
    if min_radius:
        querystring['min_radius'] = min_radius
    if min_period:
        querystring['min_period'] = min_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "planets-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

