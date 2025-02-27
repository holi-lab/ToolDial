import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def rentalizer(address: str, bathrooms: int=2, accommodates: str='2', bedrooms: int=2, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rental Property Calculator by address"
    
    """
    url = f"https://airdna1.p.rapidapi.com/rentalizer"
    querystring = {'address': address, }
    if bathrooms:
        querystring['bathrooms'] = bathrooms
    if accommodates:
        querystring['accommodates'] = accommodates
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_listings(location: str, room_types: str=None, number_of_months: int=None, bedrooms_max: int=None, accommodates_max: int=None, start_month: int=None, bedrooms_min: int=None, show_regions: bool=None, accommodates_min: int=None, currency: str='native', start_year: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rental properties of Airbnb and Vrbo by location and date."
    room_types: You can choose multiple by separating by comma.
Possible values : **(entire_home, private_room, shared_room)**.

        
    """
    url = f"https://airdna1.p.rapidapi.com/properties"
    querystring = {'location': location, }
    if room_types:
        querystring['room_types'] = room_types
    if number_of_months:
        querystring['number_of_months'] = number_of_months
    if bedrooms_max:
        querystring['bedrooms_max'] = bedrooms_max
    if accommodates_max:
        querystring['accommodates_max'] = accommodates_max
    if start_month:
        querystring['start_month'] = start_month
    if bedrooms_min:
        querystring['bedrooms_min'] = bedrooms_min
    if show_regions:
        querystring['show_regions'] = show_regions
    if accommodates_min:
        querystring['accommodates_min'] = accommodates_min
    if currency:
        querystring['currency'] = currency
    if start_year:
        querystring['start_year'] = start_year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_ratings(location: str, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns rental ratings"
    
    """
    url = f"https://airdna1.p.rapidapi.com/rentalratings"
    querystring = {'location': location, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suggest_region(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns suggested regions by keyword"
    
    """
    url = f"https://airdna1.p.rapidapi.com/suggestion"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def active_listings(location: str, start_year: int, start_month: int, number_of_months: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns active rentals"
    
    """
    url = f"https://airdna1.p.rapidapi.com/activelistings"
    querystring = {'location': location, 'start_year': start_year, 'start_month': start_month, 'number_of_months': number_of_months, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_property_managers(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the largest property managers"
    
    """
    url = f"https://airdna1.p.rapidapi.com/top_managers"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_amenities(location: str, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns rental amenities"
    
    """
    url = f"https://airdna1.p.rapidapi.com/rentalamenities"
    querystring = {'location': location, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_settings(location: str, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns rental settings."
    
    """
    url = f"https://airdna1.p.rapidapi.com/rentalsettings"
    querystring = {'location': location, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def overview(location: str, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns overview stats (ADR, OCC ,Revenue ..) of a location rental data"
    
    """
    url = f"https://airdna1.p.rapidapi.com/overview"
    querystring = {'location': location, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airdna1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

