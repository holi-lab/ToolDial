import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def car_rental_location(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search car rental location"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/cars/location/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_round_trip(departure_date: str, return_date: str, destination: str, origin: str, adults: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search round trip flights"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/flights/{origin}/{destination}/{departure_date}/{return_date}"
    querystring = {}
    if adults:
        querystring['adults'] = adults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_one_way(destination: str, departure_date: str, origin: str, adults: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search one way flights"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/flights/{origin}/{destination}/{departure_date}"
    querystring = {}
    if adults:
        querystring['adults'] = adults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def car_rental_prices(pickup_location_id: str, pickup_date: str, return_date: str, return_location_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search car rental prices"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/cars/{pickup_location_id}"
    querystring = {'pickup_date': pickup_date, 'return_date': return_date, }
    if return_location_id:
        querystring['return_location_id'] = return_location_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_city_search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search city id by keyword"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/hotels/city/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_city_nearby(longitude: str, latitude: str, radius: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search city id by geolocation"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/hotels/city/nearby/{latitude}/{longitude}"
    querystring = {}
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_list(city_id: int, checkin_date: str, checkout_date: str, currency: str='USD', offset: str='0', rooms: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List hotels by city ID"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/hotels/{city_id}"
    querystring = {'checkin_date': checkin_date, 'checkout_date': checkout_date, }
    if currency:
        querystring['currency'] = currency
    if offset:
        querystring['offset'] = offset
    if rooms:
        querystring['rooms'] = rooms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_details(checkin_date: str, is_id: int, checkout_date: str, rooms: int=1, currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel details"
    
    """
    url = f"https://priceline-com.p.rapidapi.com/hotel/{is_id}"
    querystring = {'checkin_date': checkin_date, 'checkout_date': checkout_date, }
    if rooms:
        querystring['rooms'] = rooms
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

