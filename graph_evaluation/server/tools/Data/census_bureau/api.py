import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def example_of_the_request_url_for_2010_census_sf1_data_that_has_an_assigned_key_inserted(get: str, for: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data are accessible to software developers through a stateless HTTP GET request. Up to 50 variables can be requested with a single API call. In order to access data, you will need to insert a key into the request URL in order for query results to be returned. If you do not have a key assigned to you, you will be directed to the Request a Key page where you can register for one."
    
    """
    url = f"https://community-census-bureau.p.rapidapi.com/2010/sf1"
    querystring = {'get': get, 'for': for, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-census-bureau.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def example_of_the_request_url_for_acs_2010_5_year_data_for_total_population_for_california_and_new_york(get: str, for: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-census-bureau.p.rapidapi.com/2010/acs5"
    querystring = {'get': get, 'for': for, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-census-bureau.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def example_of_the_request_url_for_acs_2011_5_year_data_for_gross_rent_as_a_percentage_of_household_income_10_0_to_14_9_percent_for_all_counties_in_ca(get: str, for: str, in: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-census-bureau.p.rapidapi.com/2011/acs5"
    querystring = {'get': get, 'for': for, 'in': in, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-census-bureau.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

