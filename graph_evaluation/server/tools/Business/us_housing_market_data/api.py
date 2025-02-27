import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def county_enriched_data(county_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for county housing market economic data. Returns data from Federal Reserve Economic Database (FRED) and Redfin."
    county_id: List of valid county_ids can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getCountyEnriched"
    querystring = {'county_id': county_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zipcode_enriched_data(zipcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for zipcode housing market economic data. Returns data from Census and Redfin."
    zipcode: List of valid zipcodes can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getZipcodeEnriched"
    querystring = {'zipcode': zipcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def region_ids(region_type: str='metro', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for region IDs for the housing market economic data."
    region_type: Valid region_types include \"metro\", \"state\", \"county\", \"city\", \"zipcode\", and \"neighborhood\"
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getRegionId"
    querystring = {}
    if region_type:
        querystring['region_type'] = region_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def neighborhood_data(neighborhood_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for neighborhood housing market economic data."
    neighborhood_id: List of valid neighborhood_ids can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getNeighborhood"
    querystring = {'neighborhood_id': neighborhood_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zipcode_data(zipcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for zipcode housing market economic data."
    zipcode: List of valid zipcodes can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getZipcode"
    querystring = {'zipcode': zipcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_data(city_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for city housing market economic data."
    city_id: List of valid city_ids can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getCity"
    querystring = {'city_id': city_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def county_data(county_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for county housing market economic data."
    county_id: List of valid county_ids can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getCounty"
    querystring = {'county_id': county_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_data(state_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for state housing market economic data."
    state_code: List of valid state_codes can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getState"
    querystring = {'state_code': state_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def metro_data(metro_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for metro housing market economic data."
    metro_id: List of valid metro_ids can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getMetro"
    querystring = {'metro_id': metro_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def national_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for national housing market economic data."
    
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getNational"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def national_enriched_data(historical_data: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for national housing market economic data. Returns data from Federal Reserve Economic Database (FRED) and Redfin."
    
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getNationalEnriched"
    querystring = {}
    if historical_data:
        querystring['historical_data'] = historical_data
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_enriched_data(state_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for state housing market economic data. Returns data from Federal Reserve Economic Database (FRED) and Redfin."
    state_code: List of valid state_ids can be found by querying the **Region Ids** endpoint
        
    """
    url = f"https://us-housing-market-data.p.rapidapi.com/getStateEnriched"
    querystring = {'state_code': state_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-housing-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

