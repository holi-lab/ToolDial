import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_one_job(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Open To Work Remote - API
		
		## Get One Job:
		
		This endpoint allows you to get all the info about a job using the job id."
    
    """
    url = f"https://open-to-work-remote-api.p.rapidapi.com/jobs/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-to-work-remote-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jobs(salarymin: str='100', company: str='SpaceX', title: str='PHP', location: str='remote', page: str='1', perpage: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Open To Work Remote - API
		
		## Search Jobs:
		
		This endpoint allows you to search for jobs.
		
		You can search for jobs by filtering by title, tags, salary, location, and company name. All the information comes paginated.
		
		*[Click 3 Dots Button To View More Details About All Params]*
		
		### Parameters and Filters:
		
		**page:** Select the page.
		**perPage:** Quantity of jobs shown per page.
		**title:** Allows search jobs by title/role.
		**tag:** Allows search jobs by tags/categories.
		**location:** Allows search jobs by location.
		**company:** Allows search jobs by the company name.
		**salaryMin:** Allows search jobs by salary.
		**source:** Allows search jobs by the Job Board source."
    
    """
    url = f"https://open-to-work-remote-api.p.rapidapi.com/jobs"
    querystring = {}
    if salarymin:
        querystring['salaryMin'] = salarymin
    if company:
        querystring['company'] = company
    if title:
        querystring['title'] = title
    if location:
        querystring['location'] = location
    if page:
        querystring['page'] = page
    if perpage:
        querystring['perPage'] = perpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-to-work-remote-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

