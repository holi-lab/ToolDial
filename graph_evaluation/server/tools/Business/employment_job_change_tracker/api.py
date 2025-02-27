import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_employment(linkedin_company_profile_url: str, linkedin_person_profile_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "track job changes"
    
    """
    url = f"https://employment-job-change-tracker.p.rapidapi.com/"
    querystring = {'LinkedIn_COMPANY_Profile_URL': linkedin_company_profile_url, 'LinkedIn_PERSON_Profile_URL': linkedin_person_profile_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "employment-job-change-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

