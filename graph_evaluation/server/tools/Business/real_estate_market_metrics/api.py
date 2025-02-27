import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def information_on_states(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all states available with their name and region code. The region code can be used as a parameter in state data requests."
    
    """
    url = f"https://real-estate-market-metrics.p.rapidapi.com/info/msa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-market-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_estate_data_on_state(state_id: str, year: str='2020', month: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns a collection of metrics and indicators by state. Providing `year` and `month` query parameters allows you to retrieve a specific month. Not providing parameters will retrieve the most recent month record.
		
		Fields:
		- year
		- month
		- month_number
		- region
		- code
		- median_listing_price
		- active_listing_count
		- median_days_on_market
		- new_listing_count
		- house_price_increased_count
		- house_price_reduced_count
		- pending_listing_count
		- median_listing_price_sq_ft
		- median_home_size_sq_ft
		- average_listing_price"
    
    """
    url = f"https://real-estate-market-metrics.p.rapidapi.com/data/state/{state_id}"
    querystring = {}
    if year:
        querystring['year'] = year
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-market-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_estate_data_on_msa(msa_id: str, year: str='2020', month: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns a collection of metrics and indicators by state. Providing `year` and `month` query parameters allows you to retrieve a specific month. Not providing parameters will retrieve the most recent month record.
		
		Fields:
		
		- year
		- month
		- month_number
		- region
		- code
		- median_listing_price
		- active_listing_count
		- median_days_on_market
		- new_listing_count
		- house_price_increased_count
		- house_price_reduced_count
		- pending_listing_count
		- median_listing_price_sq_ft
		- median_home_size_sq_ft
		- average_listing_price
		- hotness_rank
		- nielsen_household_rank
		- demand_score
		- supply_score
		- hotness_score"
    
    """
    url = f"https://real-estate-market-metrics.p.rapidapi.com/data/msa/{msa_id}"
    querystring = {}
    if year:
        querystring['year'] = year
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-market-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_estate_data_on_county(county_id: str, month: str='1', year: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns a collection of metrics and indicators by county. Providing `year` and `month` query parameters allows you to retrieve a specific month. Not providing parameters will retrieve the most recent month record.
		
		Fields:
		- year
		- month
		- month_number
		- region
		- code
		- median_listing_price
		- active_listing_count
		- median_days_on_market
		- new_listing_count
		- house_price_increased_count
		- house_price_reduced_count
		- pending_listing_count
		- median_listing_price_sq_ft
		- median_home_size_sq_ft
		- average_listing_price
		- total_listing_count
		- hotness_rank
		- nielsen_household_rank
		- demand_score
		- supply_score
		- hotness_score"
    
    """
    url = f"https://real-estate-market-metrics.p.rapidapi.com/data/county/{county_id}"
    querystring = {}
    if month:
        querystring['month'] = month
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-market-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def information_on_metropolitan_statistical_areas_msa(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all MSAs available with their name and region code. The region code can be used as a parameter in MSA data requests."
    
    """
    url = f"https://real-estate-market-metrics.p.rapidapi.com/info/msa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-market-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def information_on_counties(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all counties available with their name and region code. The region code can be used as a parameter in county data requests."
    
    """
    url = f"https://real-estate-market-metrics.p.rapidapi.com/info/county"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate-market-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

