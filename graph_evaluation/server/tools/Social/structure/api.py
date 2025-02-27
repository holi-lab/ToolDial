import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_employees(is_id: str, per_page: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a list of employees for a company"
    id: ID of the company
        per_page: Number of results per page (0-100)
        offset: The offset number to start at
        
    """
    url = f"https://structure.p.rapidapi.com/companies/{is_id}/employees"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "structure.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_jobs(is_id: str, offset: str=None, per_page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a list of jobs for a company"
    id: ID of the company
        offset: The offset number to start at
        per_page: Number of results per page (0-100)
        
    """
    url = f"https://structure.p.rapidapi.com/companies/{is_id}/jobs"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "structure.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def enrich_person(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enrich a person profile"
    
    """
    url = f"https://structure.p.rapidapi.com/people/{is_id}/enrich"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "structure.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def enrich_company(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enrich a company profile based on linkedIn key."
    
    """
    url = f"https://structure.p.rapidapi.com/companies/{is_id}/enrich"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "structure.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

