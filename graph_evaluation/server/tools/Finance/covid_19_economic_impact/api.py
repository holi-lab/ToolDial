import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def united_states_covid_19_case_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cumulative United States COVID-19 case count daily."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/covid-19-case-count-cumulative-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_covid_19_death_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cumulative United States COVID-19 death count daily."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/covid-19-death-count-cumulative-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_covid_19_new_case_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get United States COVID-19 new case count daily."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/covid-19-case-count-new-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_covid_19_new_death_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get United States COVID-19 new death count daily."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/covid-19-death-count-new-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_covid_19_new_test_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get United States COVID-19 new test count daily."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/covid-19-test-count-new-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_continued_claims_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly count of continued claims, combining Regular, PUA and PEUC claims."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/continued-claims-Regular-PUA-PEUC-us-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_employment(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily average percent change of employment relative to January 4 - 31, 2020."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/employment-Paychex-Intuit-Earnin-Kronos-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_job_postings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly average percent change of job postings relative to January 4 - 31, 2020."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/job-postings-us-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_retail_and_recreation_mobility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States percentage change in time spent at retail and recreation."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/mobility-retail-recreation-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_small_businesses_open(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States 7 day moving average percentage change of small businesses open seasonally adjusted relative to January 4 - 31, 2020."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/small-biz-open-pct-change-7-day-us-sa-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_small_businesses_revenue(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States 7 day moving average percentage change of small businesses revenue seasonally adjusted relative to January 4 - 31, 2020."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/small-biz-rev-pct-change-7-day-us-sa-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_away_from_home_mobility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States percentage change in time spent away from home."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/mobility-away-from-home-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_card_spending(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States 7 day moving average percentage change in credit and debit card spending seasonally adjusted relative to January 4 - 31, 2020."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/credit-debit-card-spending-us-sa-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_grocery_and_pharmacy_mobility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States percentage change in time spent at grocery and pharmacy."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/mobility-grocery-pharmacy-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_parks_mobility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States percentage change in time spent at parks."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/mobility-parks-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_residential_mobility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States percentage change in time spent at residential."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/mobility-residential-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_transit_stations_mobility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States percentage change in time spent inside transit stations."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/mobility-transit-stations-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_work_places_mobility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily United States percentage change in time spent at work places."
    
    """
    url = f"https://covid-19-economic-impact.p.rapidapi.com/api/v1/resources/mobility-workplaces-us-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-economic-impact.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

