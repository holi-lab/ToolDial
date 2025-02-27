import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_company_by_domain(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company’s web domain, the API will return valuable data points in JSON format."
    
    """
    url = f"https://fresh-linkedin-company-data.p.rapidapi.com/get-company-by-domain"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_data_by_its_linkedin_internal_id(company_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company’s LinkedIn internal ID, the API will return valuable data points in JSON format."
    
    """
    url = f"https://fresh-linkedin-company-data.p.rapidapi.com/get-company-by-id"
    querystring = {'company_id': company_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_data_by_linkedin_url(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company’s LinkedIn URL, the API will return valuable data points in JSON format."
    
    """
    url = f"https://fresh-linkedin-company-data.p.rapidapi.com/get-company-by-linkedinurl"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_ads_count(company_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of ads the company posted on LinkedIn. 1 request = 1 credit."
    
    """
    url = f"https://fresh-linkedin-company-data.p.rapidapi.com/get-company-ads-count"
    querystring = {'company_id': company_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_jobs_count(company_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of opening jobs the company posted on LinkedIn. 1 request = 1 credit."
    
    """
    url = f"https://fresh-linkedin-company-data.p.rapidapi.com/get-company-jobs-count"
    querystring = {'company_id': company_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

