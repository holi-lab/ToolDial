import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This provides the ability to search by ID. The ID can be retrieved by using the other endpoints."
    
    """
    url = f"https://motokaug.p.rapidapi.com/searchbyid/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motokaug.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tax_value_by_model_make_year(make: str, model: str, year: int, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    make: Make/Brand of vehicle e.g. BMW
        model: Model of vehicle e.g 1 series
        year: Year of Manufacture e.g. 2007
        type: Possible Value: 'full' or 'quick'. If Type = 'quick', response only includes tax valuation. If Type = 'full' then response also includes import duty, environmental levy, infrastructure levy, VAT and with-holding tax
        
    """
    url = f"https://motokaug.p.rapidapi.com/taxvaluebyyear/{make}/{model}/{year}/{type}"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motokaug.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tax_value_by_model_make_year_trim(make: str, model: str, trim: str, year: int, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    make: Make/Brand of vehicle e.g. BMW
        model: Model of vehicle e.g 1 series
        trim: Trim of vehicle e.g. ABA-UD20 or 2L
        year: Year of Manufacture e.g. 2007
        type: Possible Value: 'full' or 'quick'. If Type = 'quick', response only includes tax valuation. If Type = 'full' then response also includes import duty, environmental levy, infrastructure levy, VAT and with-holding tax
        
    """
    url = f"https://motokaug.p.rapidapi.com/taxvaluebytrim/{make}/{model}/{year}/{trim}/{type}"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motokaug.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tax_valuation_by_search_term(searchtext: str, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve tax valuation based on search term"
    searchtext: search term to be identified e.g. Nissan Murano 2003
        type: Possible Value: 'full' or 'quick'. If Type = 'quick', response only includes tax valuation. If Type = 'full' then response also includes import duty, environmental levy, infrastructure levy, VAT and with-holding tax
        
    """
    url = f"https://motokaug.p.rapidapi.com/search/{searchtext}/{type}"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motokaug.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_models(make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all models related to a specified make/brand of vehicle"
    make: Make/Brand of vehicle
        
    """
    url = f"https://motokaug.p.rapidapi.com/model/{make}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motokaug.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_makes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all makes listed"
    
    """
    url = f"https://motokaug.p.rapidapi.com/make"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motokaug.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

