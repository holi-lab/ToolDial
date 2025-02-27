import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nomis_authority_search(postcode: str, redirect: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the URL for a supplied postcode NOMIS local authority."
    
    """
    url = f"https://uk-property.p.rapidapi.com/nomis/generate-url"
    querystring = {'postcode': postcode, }
    if redirect:
        querystring['redirect'] = redirect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-property.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def planning_application_authority_search(postcode: str, redirect: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the URL for a supplied postcode local authority."
    
    """
    url = f"https://uk-property.p.rapidapi.com/ukGovApi/planning-application-search"
    querystring = {'postcode': postcode, }
    if redirect:
        querystring['redirect'] = redirect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-property.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domestic_epc_certificate(postcode: str, redirect: bool=None, housenumber: str='488', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the URL for a specified property's EPC rating."
    
    """
    url = f"https://uk-property.p.rapidapi.com/ukGovApi/domestic-epc-certificate"
    querystring = {'postcode': postcode, }
    if redirect:
        querystring['redirect'] = redirect
    if housenumber:
        querystring['houseNumber'] = housenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-property.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domestic_epc_search(compliant: bool, longitude: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a maximum of the 10 closest properties to the supplied latitude and longitude.
		
		Compliant properties are considered to be EPC 'C' and above. This is inline with the UK Government's proposal to force let properties to have a minimum EPC rating of 'C'."
    
    """
    url = f"https://uk-property.p.rapidapi.com/ukGovApi/domestic-epc-search"
    querystring = {'compliant': compliant, 'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-property.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

