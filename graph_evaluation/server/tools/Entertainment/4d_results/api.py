import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_draw_dates_2023(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of draw dates in 2023."
    
    """
    url = f"https://4d-results.p.rapidapi.com/get_draw_dates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_past_results_1_year(start: str, end: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all company 4D Results within a specific date range.
		(Up to 1 year data and only available for ULTRA PLAN subscriber)"
    
    """
    url = f"https://4d-results.p.rapidapi.com/get_4d_result_range/{start}/{end}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_past_results_10_years(start: str, end: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all company 4D Results within a specific date range.
		(Up to 10 year data and only available for MEGA PLAN subscriber)"
    start: Start date
        end: End date
        
    """
    url = f"https://4d-results.p.rapidapi.com/get_4d_past_results/{start}/{end}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_4d_results(version: str, date: str, company: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 4D results by date and company.
		Results including 4D and jackpot."
    company: Magnum (MAGNUM), Sports Toto (TOTO), Damacai (DAMACAI), Cashsweep (CASHSWEEP), Sabah 88 (SABAH88), Sadakan (STC), Singapore (SG)
        
    """
    url = f"https://4d-results.p.rapidapi.com/get_4d_results/{version}/{date}/{company}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_4d_dream_dictionary(digit: str, language: str, keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return relevant 4D/3D lucky draw number based on keyword (english/chinese) of your dream."
    digit: Search for 3D or 4D
        language: en for English, zh for Chinese (simplifed)
        keyword: Keyword to search for.
        
    """
    url = f"https://4d-results.p.rapidapi.com/get_dream_number/{digit}/{language}/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_4d_company_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return available company code and name.
		Code can use as parameter to query the Get 4D Results method."
    
    """
    url = f"https://4d-results.p.rapidapi.com/get_4d_companies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

