import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_job_openings(company: str=None, skip: str='0', limit: str='10', keyword: str='Marketing', location: str='California', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a way for developers to search for online jobs using an API. By sending a GET request to the endpoint with the appropriate query parameters, the API returns a list of job postings that match the search query."
    
    """
    url = f"https://job-opening-analyzer.p.rapidapi.com/jobs"
    querystring = {}
    if company:
        querystring['company'] = company
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if keyword:
        querystring['keyword'] = keyword
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "job-opening-analyzer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_most_required_skills(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the most required skills related to a job titles or a skill."
    
    """
    url = f"https://job-opening-analyzer.p.rapidapi.com/most-required-skills"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "job-opening-analyzer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

