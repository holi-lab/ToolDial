import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_employees(customerid: str, searchstring: str='Search Name and Email', limit: str='Response Count', statusfilter: str='Employee Status Filter', skip: str='Skip Count for Pagination', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Employees for a specific customer"
    
    """
    url = f"https://hris.p.rapidapi.com/employees/{customerid}"
    querystring = {}
    if searchstring:
        querystring['searchString'] = searchstring
    if limit:
        querystring['limit'] = limit
    if statusfilter:
        querystring['statusFilter'] = statusfilter
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hris.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_access_settings(customerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stored Access Setting for a specific customer"
    
    """
    url = f"https://hris.p.rapidapi.com/accessSettings/{customerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hris.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health_check(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check API Health Status"
    
    """
    url = f"https://hris.p.rapidapi.com/healthCheck"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hris.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

