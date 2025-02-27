import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def single_eu_vat_rate(country_code: str=None, ip_address: str=None, use_client_ip: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the "rate" endpoint, you may request the API to return standard and reduced VAT rates for a EU member state you specify. Important: Only one of the following three parameters for defining the country is allowed."
    country_code: Option 1 - define country by 2-letter country code
        ip_address: Option 2 - define country by IP address
        use_client_ip: Option 3 - set to "1" in order to use the IP of the client making the API request
        
    """
    url = f"https://apilayer-vatlayer-v1.p.rapidapi.com/rate"
    querystring = {}
    if country_code:
        querystring['country_code'] = country_code
    if ip_address:
        querystring['ip_address'] = ip_address
    if use_client_ip:
        querystring['use_client_ip'] = use_client_ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-vatlayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_all_eu_vat_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the API's "rate_list" endpoint to obtain standard & reduced VAT rates for all 28 current member states"
    
    """
    url = f"https://apilayer-vatlayer-v1.p.rapidapi.com/rate_list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-vatlayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_list_of_types_of_goods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request an entire list of all available "types of goods", which fall into reduced VAT categories in specific EU member states"
    
    """
    url = f"https://apilayer-vatlayer-v1.p.rapidapi.com/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-vatlayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_vat_number(vat_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the "validate" endpoint, you may request the API to validate any EU VAT number and obtain information about the company it is assigned to"
    vat_number: "vat_number" parameter - append the VAT number you want to validate
        
    """
    url = f"https://apilayer-vatlayer-v1.p.rapidapi.com/validate"
    querystring = {'vat_number': vat_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-vatlayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

