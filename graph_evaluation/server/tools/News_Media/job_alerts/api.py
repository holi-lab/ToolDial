import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def job_details(guid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API End point provides the details of selected job by its Guid Id"
    
    """
    url = f"https://job-alerts.p.rapidapi.com/jobdetails"
    querystring = {'guid': guid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "job-alerts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobslist(pageno: int, jobtype: str=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end point provides list of jobs list with page number as required parameter and keywords and job type as optional parameter"
    
    """
    url = f"https://job-alerts.p.rapidapi.com/jobs"
    querystring = {'pageno': pageno, }
    if jobtype:
        querystring['jobtype'] = jobtype
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "job-alerts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

