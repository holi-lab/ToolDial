import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def job_salary(job_title: str, location: str, radius: str='200', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get estimated job salaries/pay by job title and location."
    job_title: Job title for which to get salary estimation
        location: Free-text location/area in which to get salary estimation
        radius: Search radius in km (measured from *location*).
Default: `200`.
        
    """
    url = f"https://job-salary-data.p.rapidapi.com/job-salary"
    querystring = {'job_title': job_title, 'location': location, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "job-salary-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

