import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def postalcodepairswithinadistance(zip_code: str, country_code: str, version: str, distance: int, unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of matched postal code pairs that are within a given distance to each other."
    unit: (optional) default: 'km' (can be 'km' or 'miles')
        
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/match"
    querystring = {'zip_code': zip_code, 'country_code': country_code, 'distance': distance, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postalcodetolocationinformation(country_code: str, zip_code: int, version: str, country: str='{}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to receive information for one or multiple given postal codes."
    country: (optional) ISO 3166-1 alpha-2 country code
        
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/search"
    querystring = {'country_code': country_code, 'zip_code': zip_code, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isemaildisposable(email_id: str, version: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/is_email_disposable"
    querystring = {'email_id': email_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postalcodeswithinaradius(zip_code: int, radius: int, country_code: str, version: str, unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will convert a list of submitted postal codes into a list of postal code pairs that are located within given distance to each other."
    radius: (required) distance in 'km' or 'miles', max: 500
        unit: (optional) default: 'km' (can be 'km' or 'miles')
        
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/radius"
    querystring = {'zip_code': zip_code, 'radius': radius, 'country_code': country_code, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postalcodesbycity(city: str, country_code: str, version: str, state_name: int=50, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of postal codes based on a city.
		"
    city: (required) name of the city
        state_name: (optional) name of the province. List of provinces for a country can be retrieved using our province endpoint
        limit: (optional) limit the number of results returned
        
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/code/city"
    querystring = {'city': city, 'country_code': country_code, }
    if state_name:
        querystring['state_name'] = state_name
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountstatus(version: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def distancecalculationbetweenpostalcodes(compare: str, version: str, country_code: str, zip_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Conducts a worldwide postal code search for the code 51503"
    
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/distance"
    querystring = {'compare': compare, 'country_code': country_code, 'zip_code': zip_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validphonenumber(version: str, country_code: str, phone_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/validate_phone"
    querystring = {'country_code': country_code, 'phone_number': phone_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postalcodesbystate(country_code: str, version: str, state_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of postal codes based on a state."
    
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/code/state"
    querystring = {'country_code': country_code, 'state_code': state_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def provinces_statesofacountry(version: str, country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of postal codes based on a state."
    
    """
    url = f"https://zipcode-api-world-wide-search.p.rapidapi.com/{version}/country/province"
    querystring = {'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcode-api-world-wide-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

