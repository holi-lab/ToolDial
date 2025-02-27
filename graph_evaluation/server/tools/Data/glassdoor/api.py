import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def job_details(job_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crawl information for Glassdoor's Job."
    job_id: Use Jobs Search endpoint to retrieve IDs
        
    """
    url = f"https://glassdoor.p.rapidapi.com/job/{job_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "glassdoor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_details(company_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crawl information for Glassdoor's Companies."
    company_id: Use company search endpoint to look for IDs
        
    """
    url = f"https://glassdoor.p.rapidapi.com/company/{company_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "glassdoor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_search(location_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search a city / country / state to retrieve location_id / location_type.  Useful for jobs search"
    
    """
    url = f"https://glassdoor.p.rapidapi.com/locations/search"
    querystring = {'location_name': location_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "glassdoor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_search(company_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search companies by name."
    
    """
    url = f"https://glassdoor.p.rapidapi.com/companies/search"
    querystring = {'company_name': company_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "glassdoor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobs_search(location_id: int, location_type: str, keyword: str, page_cursor: str='AB4AAYEAHgAAAAAAAAAAAAAAAcAnhckAUgEBAQcEGhsQ16pNSro70dxiUjUW7SeKfZZSrStk5I51mXxHzCrtp1Zt1Gox6xKANcdSbluwPnLv0Lxu3SFKh8C33qAPvr9HZT+BpC/K8wtBPaUAAA==', page_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search jobs with by keyword and location."
    location_id: Glassdoor's locations are represented by ID/Type. Those tuple could be found we a locations search (see relation endpoint)
        location_type: Glassdoor's locations are represented by ID/Type. Those tuple could be found we a locations search (see relation endpoint). Here is the meaning of each value
C: CITY
N: \\\"COUNTRY
M: METRO
S: STATE
        keyword: Keyword used to search jobs, could be job title or any specific word
        page_cursor: For all page_id greater than 1, you need to specify this parameter. Each page provide the page_cursor of the next page_id in its payload
        page_id: Default page_id is 0 (first page). Each page return at most 30 jobs. You need to specify the page_cursor for all page greater than 1
        
    """
    url = f"https://glassdoor.p.rapidapi.com/jobs/search"
    querystring = {'location_id': location_id, 'location_type': location_type, 'keyword': keyword, }
    if page_cursor:
        querystring['page_cursor'] = page_cursor
    if page_id:
        querystring['page_id'] = page_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "glassdoor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

