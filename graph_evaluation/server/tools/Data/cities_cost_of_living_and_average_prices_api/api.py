import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cost_of_living_by_city(city: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve the average prices and cost of living for the desired city. Declare country and city."
    city: City name
        country: Country name
        
    """
    url = f"https://cities-cost-of-living-and-average-prices-api.p.rapidapi.com/cost_of_living"
    querystring = {'city': city, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-cost-of-living-and-average-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cost_of_living_by_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve the average cost of living in a certain country. This endpoint will calculate the average prices and cost of living across the given country."
    country: Country name to retrieve its cost of living.
        
    """
    url = f"https://cities-cost-of-living-and-average-prices-api.p.rapidapi.com/cost_of_living"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-cost-of-living-and-average-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities_by_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve all the supported cities in a given country."
    country: Country name to retrieve the cities from.
        
    """
    url = f"https://cities-cost-of-living-and-average-prices-api.p.rapidapi.com/cities"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-cost-of-living-and-average-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all the supported countries in this API."
    
    """
    url = f"https://cities-cost-of-living-and-average-prices-api.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-cost-of-living-and-average-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

