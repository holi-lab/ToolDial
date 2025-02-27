import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def states(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the average price of gas in all American states."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/usa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_prediction(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the estimated price for a given City and its expected day-to-day change (in cents). View the `/predictions-list` endpoint to view the list of valid cities."
    city: View the `/predictions-list` endpoint to view valid cities.
        
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/city-prediction"
    querystring = {'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predictions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a current prediction of tomorrow's gas price provided by Dan McTeague's [gaswizard.ca](https://gaswizard.ca/gas-price-predictions/). Returns all estimated city prices and their expected day-to-day change (in cents)."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/predictions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city(city: str, province: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the average price of gas in a specific city in Canada. You must provide both the Province and the specific City you wish to gather the price for.
		
		**Warning you must select a City and Province supported by this API**: To view the current supported locations try out the `/locations-list` endpoint to view all valid Province/City combinations."
    city: Please view the results of `/locations-list` to view the most up-to-date list of valid locations for this request.
        
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/city"
    querystring = {'city': city, 'province': province, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities(province: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the average price of gas of major cities in a specific province in Canada."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/cities"
    querystring = {'province': province, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def province(province: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the average price of gas in a specific province in Canada."
    province: A valid province name. Options include Ontario, Quebec, Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Prince Edward Island, Saskatchewan.
        
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/province"
    querystring = {'province': province, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def provinces(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the average price of gas in all Canadian provinces."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/canada"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predictions_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns valid cities that can be used for the `/city-prediction` endpoint."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/predictions-list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns full object representing collection of valid locations for this API."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/location-list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a list of valid state codes that can be used with this API."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/state-list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state(state: str, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to provide average gas price of a given American State."
    type: Type of fuel you want the price for. Default is unleaded.
        
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/state"
    querystring = {'state': state, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_list(province: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of valid cities for this API when provided a valid province. Ex: 'Ontario'"
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/city-list"
    querystring = {'province': province, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def province_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of valid provinces for this API."
    
    """
    url = f"https://canadian-gas-prices.p.rapidapi.com/province-list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "canadian-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

