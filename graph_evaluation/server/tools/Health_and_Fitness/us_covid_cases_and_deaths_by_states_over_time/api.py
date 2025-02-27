import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_covid_cases_and_deaths_by_state(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get COVID cases and deaths by state"
    
    """
    url = f"https://us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com/qSSRFV/us_covid_cases_and_deaths_by_state_/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pagination(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pagination"
    
    """
    url = f"https://us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com/qSSRFV/us_covid_cases_and_deaths_by_state_?_page=2&_limit=10"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_by_submission_date(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Filter by submission date"
    
    """
    url = f"https://us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com/qSSRFV/us_covid_cases_and_deaths_by_state_?submission_date=value"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_us_covid_cases_and_deaths(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all US COVID cases and deaths"
    
    """
    url = f"https://us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com/qSSRFV/us_covid_cases_and_deaths_by_state_"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-covid-cases-and-deaths-by-states-over-time.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

