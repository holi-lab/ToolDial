import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def age_calculator(date: str, timezone: str='US/Eastern', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates age based on the requested date. Accepts different formats and Time Zones."
    date: This endpoint will accept to following parameters formats:
YYYYMMDD
YYY-MM-DD
MMDDYYYY
MM-DD-YYYY
        timezone: Optional: Default time zone is US/Eastern. To get the correct age based on your local time you can select your timezone. Use the /timezone service to see all available time zones. 
Example: US/Eastern, US/Pacific, Europe/London
        
    """
    url = f"https://age-calculator.p.rapidapi.com/age"
    querystring = {'date': date, }
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "age-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usable_time_zones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call this endpoint to view the current endpoint. Find the appreciate time zone for your location. 
		Example: US/Eastern, US/Pacific, Europe/London ..."
    
    """
    url = f"https://age-calculator.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "age-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

