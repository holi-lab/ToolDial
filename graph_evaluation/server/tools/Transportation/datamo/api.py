import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def specs_v1_tier1(make: str, sortby: str=None, order: str=None, model: str='Model 3', page: int=None, per_page: int=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pull requested vehicle data for specific field parameters. Tier 1 allows the following to be queried by:  
		1. make
		2. model
		
		At least one query parameter is required for a successful call."
    make: At least 1 query parameter is required to make a successful call. For purpose of testing through the RapidAPI interface, this is required. Normally, only one of any additional query parameters is required. i.e. make, model, engineType, ...
        sortby: The field you would like to sort by.
        order: The sort order of the specified field.
        model: At least 1 query parameter is required to make a successful call.
        page: The page of data returned, starting with index 1 (Default 1)
        per_page: The number of entries returned per query. The default is 10 per page. The max per page is 250. 
        fields: Over 100+ returnable fields including: make, model, engineType, bodyType, msrp, etc. See the Datamo website for a full list. Leave blank to return all fields.
        
    """
    url = f"https://datamo.p.rapidapi.com/specs/v1/tier1"
    querystring = {'make': make, }
    if sortby:
        querystring['sortBy'] = sortby
    if order:
        querystring['order'] = order
    if model:
        querystring['model'] = model
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "datamo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specs_v1_tier3(make: str, bodytype: str=None, msrp: str=None, enginetype: str=None, page: int=None, search: str=None, model: str='Model 3', order: str=None, sortby: str=None, per_page: int=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pull requested vehicle data for specific field parameters. Tier 3 access grants to filter by any of the 100+ fields in the database
		
		At least one query parameter is required for a successful call."
    make: At least 1 query parameter is required to make a successful call. For purpose of testing through the RapidAPI interface, this is required. Normally, only one of any additional query parameters is required. i.e. make, model, engineType, ...
        bodytype: At least 1 query parameter is required to make a successful call.
        msrp: At least 1 query parameter is required to make a successful call.
        enginetype: At least 1 query parameter is required to make a successful call.
        page: The page of data returned, starting with index 1 (Default 1).
        search: Text search by the make, model, or year.
        model: At least 1 query parameter is required to make a successful call.
        order: The sort order of the specified field.
        sortby: The field you would like to sort by.
        per_page: The number of entries returned per query. The default is 10 per page. The max per page is 250. 
        fields: Over 100+ returnable fields including: make, model, year, engineType, bodyType, etc. See the Datamo website for a full list. Leave blank to return all fields.
        
    """
    url = f"https://datamo.p.rapidapi.com/specs/v1/tier3"
    querystring = {'make': make, }
    if bodytype:
        querystring['bodyType'] = bodytype
    if msrp:
        querystring['msrp'] = msrp
    if enginetype:
        querystring['engineType'] = enginetype
    if page:
        querystring['page'] = page
    if search:
        querystring['search'] = search
    if model:
        querystring['model'] = model
    if order:
        querystring['order'] = order
    if sortby:
        querystring['sortBy'] = sortby
    if per_page:
        querystring['per_page'] = per_page
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "datamo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specs_v1_getmakes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all the vehicle makes available, as an array of strings."
    
    """
    url = f"https://datamo.p.rapidapi.com/specs/v1/getMakes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "datamo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specs_v1_tier2(make: str, sortby: str=None, order: str=None, per_page: int=None, fields: str=None, page: int=None, model: str='Model 3', enginetype: str=None, bodytype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pull requested vehicle data for specific field parameters. Tier 2 allows the following to be queried by:  
		1. make
		2. model
		3. engineType
		4. bodyType
		
		At least one query parameter is required for a successful call."
    make: At least 1 query parameter is required to make a successful call. For purpose of testing through the RapidAPI interface, this is required. Normally, only one of any additional query parameters is required. i.e. make, model, engineType, ...
        sortby: The field you would like to sort by.
        order: The sort order of the specified field.
        per_page: The number of entries returned per query. The default is 10 per page. The max per page is 250. 
        fields: Over 100+ returnable fields including: make, model, year, engineType, bodyType, etc. See the Datamo website for a full list. Leave blank to return all fields.
        page: The page of data returned, starting with index 1 (Default 1)
        model: At least 1 query parameter is required to make a successful call.
        enginetype: At least 1 query parameter is required to make a successful call.
        bodytype: At least 1 query parameter is required to make a successful call.
        
    """
    url = f"https://datamo.p.rapidapi.com/specs/v1/tier2"
    querystring = {'make': make, }
    if sortby:
        querystring['sortBy'] = sortby
    if order:
        querystring['order'] = order
    if per_page:
        querystring['per_page'] = per_page
    if fields:
        querystring['fields'] = fields
    if page:
        querystring['page'] = page
    if model:
        querystring['model'] = model
    if enginetype:
        querystring['engineType'] = enginetype
    if bodytype:
        querystring['bodyType'] = bodytype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "datamo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

