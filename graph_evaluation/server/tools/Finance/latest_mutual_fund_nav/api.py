import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetchdividends(isin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Dividends data"
    
    """
    url = f"https://latest-mutual-fund-nav.p.rapidapi.com/fetchDividends"
    querystring = {'isin': isin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-mutual-fund-nav.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchhistoricalnav(date: str, schemecode: str=None, schemecategory: str=None, schemename: str=None, mutualfundfamily: str=None, schemetype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Historical NAV"
    date: Defind Historical Date (DD-MMM-YYYY), where MMM: Jan Feb ... Dec
Supports multiple comma separated Date
        schemecode: Define Scheme Code
Supports multiple comma separated Scheme Code
        schemecategory: Define Scheme Category
        schemename: Define Scheme Name
        mutualfundfamily: Define Mutual Fund Family
        schemetype: Define Scheme Type
        
    """
    url = f"https://latest-mutual-fund-nav.p.rapidapi.com/fetchHistoricalNAV"
    querystring = {'Date': date, }
    if schemecode:
        querystring['SchemeCode'] = schemecode
    if schemecategory:
        querystring['SchemeCategory'] = schemecategory
    if schemename:
        querystring['SchemeName'] = schemename
    if mutualfundfamily:
        querystring['MutualFundFamily'] = mutualfundfamily
    if schemetype:
        querystring['SchemeType'] = schemetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-mutual-fund-nav.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchlatestnav(mutualfundfamily: str=None, schemecode: str=None, schemename: str=None, schemetype: str=None, additional: str=None, schemecategory: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Latest NAV"
    mutualfundfamily: Define Mutual Fund Family
        schemecode: Define Scheme Code
Supports multiple comma separated Scheme Code 
        schemename: Define Scheme Name
        schemetype: Define Scheme Type
        schemecategory: Define Scheme Category
        
    """
    url = f"https://latest-mutual-fund-nav.p.rapidapi.com/fetchLatestNAV"
    querystring = {}
    if mutualfundfamily:
        querystring['MutualFundFamily'] = mutualfundfamily
    if schemecode:
        querystring['SchemeCode'] = schemecode
    if schemename:
        querystring['SchemeName'] = schemename
    if schemetype:
        querystring['SchemeType'] = schemetype
    if additional:
        querystring['Additional'] = additional
    if schemecategory:
        querystring['SchemeCategory'] = schemecategory
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-mutual-fund-nav.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallschemetypes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch All Scheme Types"
    
    """
    url = f"https://latest-mutual-fund-nav.p.rapidapi.com/fetchAllSchemeTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-mutual-fund-nav.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchschemecategoriesbyschemetype(schemetype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Scheme Categories By Scheme Type"
    schemetype: Define Scheme Type
        
    """
    url = f"https://latest-mutual-fund-nav.p.rapidapi.com/fetchSchemeCategoriesBySchemeType"
    querystring = {'SchemeType': schemetype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-mutual-fund-nav.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallschemenames(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch All Scheme Names"
    
    """
    url = f"https://latest-mutual-fund-nav.p.rapidapi.com/fetchAllSchemeNames"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-mutual-fund-nav.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallmutualfundfamilies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch All Mutual Fund Families"
    
    """
    url = f"https://latest-mutual-fund-nav.p.rapidapi.com/fetchAllMutualFundFamilies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-mutual-fund-nav.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

