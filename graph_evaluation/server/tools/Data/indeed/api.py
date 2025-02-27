import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def company_jobs(company_id: str, locality: str=None, start: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search jobs by company."
    locality: Select the Indeed's country. Each value correspond to a specific indeed subdomain. Default value if missing is 'us'
        start: Use to control the pagination of results. If omitted return the first page
        
    """
    url = f"https://indeed12.p.rapidapi.com/company/{company_id}/jobs"
    querystring = {}
    if locality:
        querystring['locality'] = locality
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indeed12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobs_search(query: str, location: str, fromage: int=3, locality: str=None, page_id: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search jobs with by query and location."
    query: Keyword used to search jobs
        fromage: Number of days.
Filter jobs that was updated between now and **fromage** days.
        locality: Select the Indeed's country. Each value correspond to a specific indeed subdomain. Default value if missing is 'us'
        page_id: Use to control the pagination of results. If omitted return the first page
        
    """
    url = f"https://indeed12.p.rapidapi.com/jobs/search"
    querystring = {'query': query, 'location': location, }
    if fromage:
        querystring['fromage'] = fromage
    if locality:
        querystring['locality'] = locality
    if page_id:
        querystring['page_id'] = page_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indeed12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_details(company_id: str, locality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crawl information for Indeed's Companies."
    locality: Select the Indeed's country. Each value correspond to a specific indeed subdomain. Default value if missing is 'us'
        
    """
    url = f"https://indeed12.p.rapidapi.com/company/{company_id}"
    querystring = {}
    if locality:
        querystring['locality'] = locality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indeed12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_search(company_name: str, locality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search companies by name."
    locality: Select the Indeed's country. Each value correspond to a specific indeed subdomain. Default value if missing is 'us'
        
    """
    url = f"https://indeed12.p.rapidapi.com/companies/search"
    querystring = {'company_name': company_name, }
    if locality:
        querystring['locality'] = locality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indeed12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job_details(job_id: str, locality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crawl information for Indeed's Job."
    job_id: You can retrieve jobs ids from on jobs searches
        locality: Default value if missing is 'us'
        
    """
    url = f"https://indeed12.p.rapidapi.com/job/{job_id}"
    querystring = {}
    if locality:
        querystring['locality'] = locality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indeed12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

