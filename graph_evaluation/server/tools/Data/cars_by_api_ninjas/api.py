import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_cars(model: str='corolla', max_city_mpg: int=None, min_comb_mpg: int=None, max_hwy_mpg: int=None, fuel_type: str=None, limit: str=None, drive: str=None, max_comb_mpg: int=None, make: str=None, transmission: str=None, year: str=None, min_hwy_mpg: int=None, min_city_mpg: int=None, cylinders: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Cars API endpoint."
    model: Model of vehicle.
        max_city_mpg: Maximum city fuel efficiency in miles per gallon.
        min_comb_mpg: Minimum combination (city + highway) fuel efficiency in miles per gallon.
        max_hwy_mpg: Maximum highway fuel efficiency in miles per gallon.
        fuel_type: Type of fuel used. Possible values: **gas**, **diesel**, **electricity**
        limit: How many results to return. Must be between **1** and **30**. Default is **5**
        drive: Drive transmission. Possible values: **fwd** (front-wheel drive), **rwd** (rear-wheel drive), **awd** (all-wheel drive), **4wd** (four-wheel drive)
        max_comb_mpg: Maximum combination (city + highway) fuel efficiency in miles per gallon.
        make: Vehicle manufacturer.
        transmission: Type of transmission. Possible values: **manual**, **automatic**
        year: Vehicle model year.
        min_hwy_mpg: Minimum highway fuel efficiency in miles per gallon.
        min_city_mpg: Minimum City fuel efficiency in miles per gallon.
        cylinders: Number of cylinders. Possible values: **2, 3 4, 5, 6, 8, 10, 12, 16**
        
    """
    url = f"https://cars-by-api-ninjas.p.rapidapi.com/v1/cars"
    querystring = {}
    if model:
        querystring['model'] = model
    if max_city_mpg:
        querystring['max_city_mpg'] = max_city_mpg
    if min_comb_mpg:
        querystring['min_comb_mpg'] = min_comb_mpg
    if max_hwy_mpg:
        querystring['max_hwy_mpg'] = max_hwy_mpg
    if fuel_type:
        querystring['fuel_type'] = fuel_type
    if limit:
        querystring['limit'] = limit
    if drive:
        querystring['drive'] = drive
    if max_comb_mpg:
        querystring['max_comb_mpg'] = max_comb_mpg
    if make:
        querystring['make'] = make
    if transmission:
        querystring['transmission'] = transmission
    if year:
        querystring['year'] = year
    if min_hwy_mpg:
        querystring['min_hwy_mpg'] = min_hwy_mpg
    if min_city_mpg:
        querystring['min_city_mpg'] = min_city_mpg
    if cylinders:
        querystring['cylinders'] = cylinders
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cars-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

