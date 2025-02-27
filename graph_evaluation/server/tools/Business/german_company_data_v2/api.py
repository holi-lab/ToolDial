import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def companies_id_events(is_id: str, category: str='MANAGEMENT_AND_TEAM,FINANCES_AND_CAPITAL,NEWS_AND_EVENTS', since: str='2017-01-01', size: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the latest events about the company with the given identifier since the given timestamp. This timestamp is optional and is specified via a query parameter. Events come with a type (companies' register event, blog article, press release, etc.), a category (change in management, finances, merger/acquisition, etc.), a text (the content of the event), a source, and a timestamp."
    
    """
    url = f"https://german-company-data.p.rapidapi.com/companies/{is_id}/events"
    querystring = {}
    if category:
        querystring['category'] = category
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_id_financials(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the financial key figures (balance sheet total, revenue, profit, number of employees) for the company with the given identifier. For each key figure the endpoint returns a list of values associated with the corresponding year."
    
    """
    url = f"https://german-company-data.p.rapidapi.com/companies/{is_id}/financials"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_id_financials_exists(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the list of years for which financial key figures (balance sheet total, revenue, profit, number of employees) for the company with the given identifier are available. The purpose of this endpoint is to give the user the opportunity to avoid potentially expensive and useless calles to the /financials-endpoint."
    
    """
    url = f"https://german-company-data.p.rapidapi.com/companies/{is_id}/financials/exists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_id_jobs(is_id: str, since: str='2017-01-01', size: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the latest job postings about the company with the given identifier since the given timestamp. This timestamp is optional and is specified via a query parameter. Job postings come with a title, a text (the content of the posting), a source, and a timestamp."
    
    """
    url = f"https://german-company-data.p.rapidapi.com/companies/{is_id}/jobs"
    querystring = {}
    if since:
        querystring['since'] = since
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_id_data(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the basic information of the company with the given identifier. The information includes all the core data (name, address), web data if known (phone, fax, email, url, social media profiles), legal information (registry, founding date, capital), up to three industries, and revenue and employees class."
    id: The Implisense identifier of the company.
        
    """
    url = f"https://german-company-data.p.rapidapi.com/companies/{is_id}/data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_id_people(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the information about people associated with the company with the given identifier. The information includes names, roles, email addresses, and phone numbers, if available."
    
    """
    url = f"https://german-company-data.p.rapidapi.com/companies/{is_id}/people"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

