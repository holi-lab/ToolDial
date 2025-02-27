import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_registration_number(regnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search company information by the company or business' registration number issued by the CAC. It returns a single match if the registration number exists."
    
    """
    url = f"https://business-and-company-name-api.p.rapidapi.com/api/v1/cacctrlsrvc/companies/searchbyreg"
    querystring = {'regNumber': regnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-and-company-name-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_company_by_name(page: int, limit: int, companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Do you know the company name? you can make a search of company information with just the name. You will get hits of all companies and businesses that are matching that particular name."
    
    """
    url = f"https://business-and-company-name-api.p.rapidapi.com/api/v1/cacctrlsrvc/companies/searchbyname"
    querystring = {'page': page, 'limit': limit, 'companyName': companyname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-and-company-name-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_companies_paginated(page: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets all the companies and business as in the CAC database."
    
    """
    url = f"https://business-and-company-name-api.p.rapidapi.com/api/v1/cacctrlsrvc/companies/all"
    querystring = {'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-and-company-name-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

