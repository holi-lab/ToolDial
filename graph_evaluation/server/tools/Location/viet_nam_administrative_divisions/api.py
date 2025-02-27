import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_wards_of_a_district_in_vietnam(district: str='001', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all names of wards in a district in Vietnam"
    district: It is district code got from API /districts
It not set, return all wards
        
    """
    url = f"https://viet-nam-administrative-divisions.p.rapidapi.com/wards"
    querystring = {}
    if district:
        querystring['district'] = district
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "viet-nam-administrative-divisions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_districts_of_a_city_in_vietnam(city: str='01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all names of districts in a city"
    city: It is a city code got from API /cities.
If not set, it will get all districts
        
    """
    url = f"https://viet-nam-administrative-divisions.p.rapidapi.com/districts"
    querystring = {}
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "viet-nam-administrative-divisions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_cities_in_vietnam(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all names of cities in Vietnam"
    
    """
    url = f"https://viet-nam-administrative-divisions.p.rapidapi.com/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "viet-nam-administrative-divisions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

