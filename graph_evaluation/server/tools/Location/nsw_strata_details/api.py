import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_of_suburbs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all of the suburbs and the number of strata plans within that suburb."
    
    """
    url = f"https://nsw-strata-details.p.rapidapi.com/suburbs/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nsw-strata-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def strata_plan_details(plan: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single strata plan including 2021 Australian Census dervied data  for both the suburb and local government areas."
    
    """
    url = f"https://nsw-strata-details.p.rapidapi.com/"
    querystring = {'plan': plan, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nsw-strata-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def strata_plans_within_a_local_government_area_lga(lga: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all of the strata plans that are within a specific local government area including 2021 Australian Census dervied data ."
    
    """
    url = f"https://nsw-strata-details.p.rapidapi.com/"
    querystring = {'lga': lga, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nsw-strata-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_local_government_areas(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all of the local government areas  and the number of strata plans within that local government area."
    
    """
    url = f"https://nsw-strata-details.p.rapidapi.com/lgas/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nsw-strata-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def strata_plans_within_a_suburb(suburb: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all of the strata plans that are within a specific suburb including 2021 Australian Census dervied data ."
    suburb: see the `/suburbs/` call for a list of all of the suburbs and the number of strata plans.
        page: If a search spans more than one page, then this will paginate through the result.

The current page number and number of pages will be returned in the `response` JSON Object with the keys:

 - `page` - the current page number
 - `num_pages` - the number of pages within the result set


        
    """
    url = f"https://nsw-strata-details.p.rapidapi.com/"
    querystring = {'suburb': suburb, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nsw-strata-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

