import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_location(checkout: str, checkin: str, location: str, adults: int, pets: int=0, currency: str='USD', page: int=1, infants: int=0, children: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find rooms at a location"
    checkout: Check-out date
        checkin: Check-in date
        location: Location (city, region, suburb, etc.)
        adults: Number of adult guests (13 years and over)
        pets: Number of pets
        currency: Price currency, default: USD
        page: Page of results returned
Allowed values: 1-8
        infants: Number of infants (under 2 years)
        children: Number of children (2-12 years)
        
    """
    url = f"https://airbnb13.p.rapidapi.com/search-location"
    querystring = {'checkout': checkout, 'checkin': checkin, 'location': location, 'adults': adults, }
    if pets:
        querystring['pets'] = pets
    if currency:
        querystring['currency'] = currency
    if page:
        querystring['page'] = page
    if infants:
        querystring['infants'] = infants
    if children:
        querystring['children'] = children
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendar(room_id: str, month: str='12', count: str='1', year: str='2023', currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns availability and prices of a listing"
    month: Calendar month, default: current month
        count: Returned number of months, default: 12, max: 12
        year: Calendar year, default: current year
        currency: Price currency, default: USD
        
    """
    url = f"https://airbnb13.p.rapidapi.com/calendar"
    querystring = {'room_id': room_id, }
    if month:
        querystring['month'] = month
    if count:
        querystring['count'] = count
    if year:
        querystring['year'] = year
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_geo_coordinates(checkin: str, sw_lat: int, sw_lng: int, ne_lng: int, adults: int, checkout: str, ne_lat: int, pets: int=0, children: int=0, page: int=1, currency: str='USD', infants: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find rooms in a rectangles bounded by provided coordinates"
    checkin: Check-in date
        sw_lat: Latitude of the southwestern corner of the search area
        sw_lng: Longitude of the southwestern corner of the search area
        ne_lng: Longitude of the northeastern  corner of the search area
        adults: Number of adult guests (13 years and over)
        checkout: Check-out date
        ne_lat: Latitude of the northeastern  corner of the search area
        pets: Number of pets
        children: Number of children (2-12 years)
        page: Returned page of results.
Allowed values: 1-8
        currency: Price currency, default: USD
        infants: Number of infants (under 2 years)
        
    """
    url = f"https://airbnb13.p.rapidapi.com/search-geo"
    querystring = {'checkin': checkin, 'sw_lat': sw_lat, 'sw_lng': sw_lng, 'ne_lng': ne_lng, 'adults': adults, 'checkout': checkout, 'ne_lat': ne_lat, }
    if pets:
        querystring['pets'] = pets
    if children:
        querystring['children'] = children
    if page:
        querystring['page'] = page
    if currency:
        querystring['currency'] = currency
    if infants:
        querystring['infants'] = infants
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find locations by query"
    query: Query parameter
        
    """
    url = f"https://airbnb13.p.rapidapi.com/autocomplete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

