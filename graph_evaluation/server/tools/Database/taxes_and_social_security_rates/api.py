import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_social_security_rate_for_employees_by_continent(taxsseid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns Get Social Security Rate For Employees by Continent or Group"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesSSE/{taxsseid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_social_security_rate_for_employees(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns All Social Security Rate For Employees"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesSSE"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_social_security_rate_for_companies_by_continent(taxsscid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns Social Security Rate For Companies by Continent"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesSSC/{taxsscid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_social_security_rate_for_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns all the Social Security Rate For Companies"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesSSC"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def social_security_rate_by_continent(taxssid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns Security Rate by Continent or Group"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesSS/{taxssid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_social_security_rate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns all the Social Security Rate around the World."
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesSS"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_sales_tax_rate_by_continent(taxsid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns all Sales Tax Rate by Continent/Group"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesS/{taxsid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_sales_tax_rate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns all the All Sales Tax Rate around the World."
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesS"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_personal_income_tax_rate_by_continent(taxpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns the Personal Income Tax Rate by Continent"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesP/{taxpid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_personal_income_tax_rate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns all the Personal Income Tax Rate around the Globe"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesP"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tax_rates_by_continent(taxcid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns all the tax rate per Continent or Group of Country"
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesC/{taxcid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_corporate_tax_rate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns all the tax rate around the World."
    
    """
    url = f"https://taxes-and-social-security-rates.p.rapidapi.com/taxesC"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxes-and-social-security-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

