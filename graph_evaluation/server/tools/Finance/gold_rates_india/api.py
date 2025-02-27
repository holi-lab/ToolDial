import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_city_gold_prices_22k_and_24k(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Gold 22 K and 24K rates by the major city Names by this endpoint."
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Gold-City/"
    querystring = {'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def silver_prices_by_state_names(state: str='uttar-pradesh', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Silver Prices By State Names of India"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Silver-State/"
    querystring = {}
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def diamond_price_history_in_indian_cities(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Diamond Price History in Indian Cities"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Diamond-City-History/"
    querystring = {'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def diamond_prices_in_indian_cities(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Diamond Prices in Indian Cities"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Diamond-City/"
    querystring = {'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def diamond_prices_in_indian_states(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Diamond Prices In Indian States"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Diamond-State/"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_gold_history_by_city(city: str='noida', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Gold History By City"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Gold-City-History/"
    querystring = {}
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gold_price_india_history(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the History of Gold Prices from Previous Week.
		
		
		tf:  Price of 24Carat Gold
		tt: Price of 22Carat Gold"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Gold-history/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platinum_prices_historical_data_by_city_names(city: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Platinum Prices Historical Data By City Names"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Platinum-City-History/"
    querystring = {'city': city, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platinum_prices_by_indian_cities(city: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Platinum Prices in All Major Cities in India"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Platinum-City/"
    querystring = {'city': city, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platinum_prices_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Platinum Prices in INR, Dollar and Pound by All Indian States"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Platinum-State/"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def silver_historical_prices_by_city(city: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Historical Data for Silver Prices by City Names"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Silver-City-History/"
    querystring = {'city': city, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def silver_prices_by_city_names(state: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Silver Prices By City Names"
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Silver-City/"
    querystring = {'state': state, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def silver(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest Silver Rates for all Indian Cities.  Rates are available for 1gm and 1Kg Silver."
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Silver/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gold_prices_24k_and_22k_major_cities_in_india(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is New API for Gold Rates in India with Most of the state and cities data.
		
		1Gram22K => Rate of 1 Gram Gold of 22 K in INR
		8Gram22K => Rate of 8 Gram Gold of 22 K in INR
		1Gram24K => Rate of 1 Gram Gold of 24 K in INR
		8Gram24K => Rate of 1 Gram Gold of 24 K in INR
		
		Feel free to drop a message if you get into any trouble."
    
    """
    url = f"https://gold-rates-india.p.rapidapi.com/Gold-Temp/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-rates-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

