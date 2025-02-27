import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v2_jobs_latest(pagesize: int=None, pagenumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the latest job listing, with or without pagination."
    
    """
    url = f"https://jobsearch4.p.rapidapi.com/api/v2/Jobs/Latest"
    querystring = {}
    if pagesize:
        querystring['PageSize'] = pagesize
    if pagenumber:
        querystring['PageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jobsearch4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_jobs_slug(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a job by slug"
    
    """
    url = f"https://jobsearch4.p.rapidapi.com/api/v2/Jobs/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jobsearch4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_jobs_search(pagesize: int=12, pagenumber: int=1, searchquery: str='java', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for your dream job"
    
    """
    url = f"https://jobsearch4.p.rapidapi.com/api/v2/Jobs/Search"
    querystring = {}
    if pagesize:
        querystring['PageSize'] = pagesize
    if pagenumber:
        querystring['PageNumber'] = pagenumber
    if searchquery:
        querystring['SearchQuery'] = searchquery
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jobsearch4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

