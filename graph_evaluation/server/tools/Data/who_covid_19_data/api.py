import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of valid region."
    
    """
    url = f"https://who-covid-19-data.p.rapidapi.com/api/data/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "who-covid-19-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_data(deaths: int=None, newcases: int=None, newdeaths: int=None, transmissiontype: int=None, region: str='European Region', cases: int=35000, reportnumber: int=None, territory: bool=None, reportdate: str='2020-03-25', name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the data.
		(If no query parameters are provided, returns data for the most recent report)"
    deaths: Limit search to n number of total confirmed deaths and above.
        newcases: Limit search to n number of new confirmed cases (since the previous report) and above.
        newdeaths: Limit search to n number of new confirmed deaths (since the previous report) and above.
        transmissiontype: Filter data by a specific method of virus transmission.
        region: Enter a valid region.
        cases: Limit search to n number of total confirmed cases and above.
        reportnumber: Each report has a report number attached.
        territory: Get only countries or only territories.
        reportdate: Get data for a specific day.
        name: Enter a valid country or territory.
        
    """
    url = f"https://who-covid-19-data.p.rapidapi.com/api/data"
    querystring = {}
    if deaths:
        querystring['deaths'] = deaths
    if newcases:
        querystring['newCases'] = newcases
    if newdeaths:
        querystring['newDeaths'] = newdeaths
    if transmissiontype:
        querystring['transmissionType'] = transmissiontype
    if region:
        querystring['region'] = region
    if cases:
        querystring['cases'] = cases
    if reportnumber:
        querystring['reportNumber'] = reportnumber
    if territory:
        querystring['territory'] = territory
    if reportdate:
        querystring['reportDate'] = reportdate
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "who-covid-19-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of valid country and territory names."
    
    """
    url = f"https://who-covid-19-data.p.rapidapi.com/api/data/names"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "who-covid-19-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transmission_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Describes each transmission type."
    
    """
    url = f"https://who-covid-19-data.p.rapidapi.com/api/data/transmissionTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "who-covid-19-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available report dates."
    
    """
    url = f"https://who-covid-19-data.p.rapidapi.com/api/data/dates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "who-covid-19-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

