import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_stars(min_distance_light_year: int=None, min_apparent_magnitude: int=None, constellation: int=None, max_apparent_magnitude: int=None, offset: int=None, name: str='vega', max_absolute_magnitude: int=None, min_absolute_magnitude: int=None, max_distance_light_year: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of stars matching specified parameters. Returns at most 30 results. To access more than 30 results, use the offset parameter to offset results in multiple API calls."
    min_distance_light_year: minimum distance the star is from Earth in light years.
        min_apparent_magnitude: minimum apparent magnitude brightness of the star.
        constellation: the constellation the star belongs to.
        max_apparent_magnitude: maximum apparent magnitude brightness of the star.
        offset: number of results to offset for pagination.
        name: the name of the star. Note that many of the star names contain greek characters.
        max_absolute_magnitude: maximum absolute magnitude brightness of the star.
        min_absolute_magnitude: minimum absolute magnitude brightness of the star.
        max_distance_light_year: maximum distance the star is from Earth in light years.
        
    """
    url = f"https://stars-by-api-ninjas.p.rapidapi.com/v1/stars"
    querystring = {}
    if min_distance_light_year:
        querystring['min_distance_light_year'] = min_distance_light_year
    if min_apparent_magnitude:
        querystring['min_apparent_magnitude'] = min_apparent_magnitude
    if constellation:
        querystring['constellation'] = constellation
    if max_apparent_magnitude:
        querystring['max_apparent_magnitude'] = max_apparent_magnitude
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    if max_absolute_magnitude:
        querystring['max_absolute_magnitude'] = max_absolute_magnitude
    if min_absolute_magnitude:
        querystring['min_absolute_magnitude'] = min_absolute_magnitude
    if max_distance_light_year:
        querystring['max_distance_light_year'] = max_distance_light_year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stars-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

