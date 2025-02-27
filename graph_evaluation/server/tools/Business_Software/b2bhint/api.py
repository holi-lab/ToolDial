import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_person_by_name(q: str, countrycode: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a person by parameters will return a list of persons that match your query"
    
    """
    url = f"https://b2bhint.p.rapidapi.com/api/v1/rapidapi/person/search"
    querystring = {'q': q, }
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "b2bhint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_full_data(internationalnumber: str, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Get company details endpoint will return all full company data, including company contacts, financial reports and other data"
    
    """
    url = f"https://b2bhint.p.rapidapi.com/api/v1/rapidapi/company/full"
    querystring = {'internationalNumber': internationalnumber, 'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "b2bhint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_company_by_name(q: str, countrycode: str='be', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a company by parameters will return a list of companies that match your query"
    q: Company name or number or other identifiers
        countrycode: ISO2 country code
        
    """
    url = f"https://b2bhint.p.rapidapi.com/api/v1/rapidapi/company/search"
    querystring = {'q': q, }
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "b2bhint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_basic_data(countrycode: str, internationalnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Get company details endpoint will return all basic company data on B2BHint."
    
    """
    url = f"https://b2bhint.p.rapidapi.com/api/v1/rapidapi/company/basic"
    querystring = {'countryCode': countrycode, 'internationalNumber': internationalnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "b2bhint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_company_by_email(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a company by email will return a list of companies that match the selected email"
    
    """
    url = f"https://b2bhint.p.rapidapi.com/api/v1/rapidapi/company/search-by-email"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "b2bhint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

