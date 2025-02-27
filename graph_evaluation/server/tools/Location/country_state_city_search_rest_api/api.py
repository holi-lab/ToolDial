import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cities_by_country_code(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with the names and codes of all cities belonging to the country specified by the "countrycode" query parameter. We recommend using the "Cities By Country Code and State Code" endpoint for listing cities as it is faster and more accurate than this one."
    
    """
    url = f"https://country-state-city-search-rest-api.p.rapidapi.com/cities-by-countrycode"
    querystring = {'countrycode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-state-city-search-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_by_country_code_and_state_code(statecode: str, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with the names and codes of all cities belonging to the state corresponding to the "statecode" query parameter. It is important to note that both the "countrycode" and "statecode" query parameters are required for a successful API request."
    
    """
    url = f"https://country-state-city-search-rest-api.p.rapidapi.com/cities-by-countrycode-and-statecode"
    querystring = {'statecode': statecode, 'countrycode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-state-city-search-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def states_by_country_code(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with the names and codes of all states or provinces belonging to the country specified by the "countrycode" query parameter."
    
    """
    url = f"https://country-state-city-search-rest-api.p.rapidapi.com/states-by-countrycode"
    querystring = {'countrycode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-state-city-search-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_data_by_country_code(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with an object containing the information for the country corresponding to the country code provided in the "countrycode" query parameter. This object contains the same information as the object for that country in the "all countries" endpoint."
    
    """
    url = f"https://country-state-city-search-rest-api.p.rapidapi.com/country-data-by-countrycode"
    querystring = {'countrycode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-state-city-search-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with all the countries in the world. The countries are returned as objects from the server. These objects, in addition to the country names and ISO codes, also contain some other important information."
    
    """
    url = f"https://country-state-city-search-rest-api.p.rapidapi.com/allcountries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-state-city-search-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

