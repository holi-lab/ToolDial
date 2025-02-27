import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def expiration_date(expiration_date: str='SAMPLEDATE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the expiration date of the laboratory's certificate.  
		The request searches on the expiration_date field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if expiration_date:
        querystring['expiration_date'] = expiration_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def general_facility_type(general_facility_type: str='Hospital', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the facility type of the laboratory.  
		The request searches on the general_facility_type field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if general_facility_type:
        querystring['general_facility_type'] = general_facility_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def certification_type(type: str='Sample Accreditation', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the certification type of the laboratory.  
		The request searches on the type field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_number(phone_number: str='5555555555', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the phone number for the laboratory.  
		The request searches on the phone_number field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if phone_number:
        querystring['phone_number'] = phone_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_code(zip_code: str='0001', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the zip code of the laboratory.  
		The request searches on the zip_code field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state(state_code: str='NA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the state of the laboratory.  
		The request searches on the state_code field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if state_code:
        querystring['state_code'] = state_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city(city_name: str='SAMPLE_CITY1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the city of the laboratory.  
		The request searches on the city_name field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if city_name:
        querystring['city_name'] = city_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def address(address: str='1 SAMPLE Drive', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the address of the laboratory.  
		The request searches on the address field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def provider_number(provider_number: str='0000000001', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the provider number for the laboratory.  
		The request searches on the provider_number field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if provider_number:
        querystring['provider_number'] = provider_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facility(facility: str='SAMPLE-1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General GET request for lab demographics based on the name of the laboratory.  
		The request searches on the facility field."
    
    """
    url = f"https://clia-lab-search.p.rapidapi.com/api/labs"
    querystring = {}
    if facility:
        querystring['facility'] = facility
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "clia-lab-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

