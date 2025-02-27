import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_companies_by_name(company_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will allow you to search for companies by their name. This endpoint will provide you with the company slug that could be used to get additional company data, like SIC code and NAICS code."
    company_name: Company name to lookup.
        
    """
    url = f"https://get-companies-by-sic-code-api.p.rapidapi.com/company-name"
    querystring = {'company_name': company_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-companies-by-sic-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_data_by_company_slug(company_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the Company Slug that is provided in the "Get Companies by SIC code" endpoint, you will be receiving additional information about a company. 
		Get city, state, zip code, estimated annual revenue, years in business, and estimated company size, and company NAICS."
    company_slug: Company Slug provided in "Get Companies by SIC code" endpoint.
        
    """
    url = f"https://get-companies-by-sic-code-api.p.rapidapi.com/company-slug"
    querystring = {'company_slug': company_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-companies-by-sic-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_companies_by_sic_code(sic_code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of the top 100 companies that are related to a given SIC code."
    sic_code: SIC code to lookup.
        
    """
    url = f"https://get-companies-by-sic-code-api.p.rapidapi.com/sic-code"
    querystring = {'sic_code': sic_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-companies-by-sic-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

