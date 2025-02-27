import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_masks_usage_instructions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use these instructions as pictures to avoid infection"
    
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/random_masks_usage_instructions.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_cases(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return data about cases or coronavirus in United States by state"
    
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/united_states_cases.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_stat_by_country_iso_alpha_2(alpha2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides information about latest stat by country ISO alpha2 code. Codes list will be taken from iban site. 
		https://www.iban.com/country-codes"
    alpha2: ISO Alpha2 code. Required

Codes are from here: https://www.iban.com/country-codes
        
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/latest_stat_by_iso_alpha_2.php"
    querystring = {'alpha2': alpha2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history_by_particular_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We will return history by particular country that was provided in parameters."
    country: the country name
        
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/cases_by_particular_country.php"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def johns_hopkins_latest_usa_statistic_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides statistics by state about cases and deaths in U.S. based on Johns Hopkins"
    state: State name e.g. Washington
        
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/johns_hopkins_latest_usa_statistic_by_state.php"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stat_by_country_iso_code(iso: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest statistic by country iso code (3 letters)"
    
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/who_latest_stat_by_iso.php"
    querystring = {'iso': iso, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stat_by_country_name(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  latest country information by country name"
    
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/who_latest_stat_by_country.php"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def world_total_stat(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Total statistic for world cases, deaths, etc"
    
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/worldstat.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_stat_by_country_name(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the most recent stat by provided country name"
    country: Country name
        
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/latest_stat_by_country.php"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def affected_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of affected countries by date"
    
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/affected.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_stat_by_country_iso_alpha_3(alpha3: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides information about latest stat by country ISO alpha3 code. Codes list will be taken from iban site.
		https://www.iban.com/country-codes"
    alpha3: Country alpha3 code taken from iban site. 
https://www.iban.com/country-codes
        
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/latest_stat_by_iso_alpha_3.php"
    querystring = {'alpha3': alpha3, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history_by_particular_country_by_date(country: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get history by country and date"
    country: Country name
        date: Date in format 2020-03-20
        
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/history_by_particular_country_by_date.php"
    querystring = {'country': country, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cases_by_country(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return JSON with cases mapped by country map"
    
    """
    url = f"https://coronavirus-monitor.p.rapidapi.com/coronavirus/cases_by_country.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coronavirus-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

