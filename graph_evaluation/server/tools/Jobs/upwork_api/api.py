import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of county regions."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/metadata/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tests(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of available tests at Upwork."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/metadata/tests"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_skills(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of skills available in a freelancer's profile."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/metadata/skills"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of categories for a job/freelancer profile."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/metadata/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_freelancer_profile_information(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns an exhaustive list of attributes associated with the freelancer."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/freelancers/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_freelancers(count: int=5, skills: str='vue', rate: str='[15 TO 40]', q: str='laravel', offset: int=0, title: str='Full Stack Develoer', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of the objects with information about each freelancer who matches the requested query and parameters. 
		At least one of the `q`, `title`, `skill` parameters should be specified. Use 'offset' and 'count' parameters for pagination. 
		'rate' parameter a number or range used to filter the search by freelancer's profile rate. Single values such as `20` or `20,30` (comma-separated values result in `OR` queries) and ranges such as `[20 TO 40]` are valid."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/freelancers"
    querystring = {}
    if count:
        querystring['count'] = count
    if skills:
        querystring['skills'] = skills
    if rate:
        querystring['rate'] = rate
    if q:
        querystring['q'] = q
    if offset:
        querystring['offset'] = offset
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_job(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns the complete job object by job id."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/jobs/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jobs(keyword: str, count: int=50, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request searching Upwork jobs by given keyword. Response value will be paginated and if you need to get next page data set the 'offset' and 'count' query parameters. By default value of 'offset' is '0' and 'count' is 50."
    
    """
    url = f"https://upwork-api2.p.rapidapi.com/jobs"
    querystring = {'keyword': keyword, }
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

