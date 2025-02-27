import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sqlsearch(sql: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For advanced users only - run a custom SQL select statement against the database; 
		
		A sample query is as follows;
		
		select * from Names n 
		join companies c on c.companyId = n.companyId
		left join positions p on p.companyId = n.companyId
		left join referenceNumbers rn on rn.companyId = n.companyId
		left join addresses a on a.companyId = n.companyId
		left join capital cp on cp.companyId = n.companyId
		where name like 'cci Dialog%'"
    
    """
    url = f"https://german-company-data1.p.rapidapi.com/openregister.json"
    querystring = {'sql': sql, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbyname(name_like: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Seach for gernan companies by name, using the % symbol as a wildcard, to obtain the company ID used in other API calls"
    
    """
    url = f"https://german-company-data1.p.rapidapi.com/openregister/Names.json"
    querystring = {'name__like': name_like, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def capitalbycompanyid(companyid_exact: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company financials by company id"
    
    """
    url = f"https://german-company-data1.p.rapidapi.com/openregister/Capital.json"
    querystring = {'companyId__exact': companyid_exact, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def positionsbycompanyid(companyid_exact: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Look up company directors by company id"
    
    """
    url = f"https://german-company-data1.p.rapidapi.com/openregister/Positions.json"
    querystring = {'companyId__exact': companyid_exact, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addressbycompanyid(companyid_exact: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company registered addresses by company ID"
    
    """
    url = f"https://german-company-data1.p.rapidapi.com/openregister/Addresses.json"
    querystring = {'companyId__exact': companyid_exact, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def objectivesbycompanyid(companyid_exact: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive the company objectives (i.e. purpose) in German for a given company"
    
    """
    url = f"https://german-company-data1.p.rapidapi.com/openregister/Objectives.json"
    querystring = {'companyId__exact': companyid_exact, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referencenumbersbycompanyid(companyid_exact: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Look up the reference numbers of a company (court register etc.) from company ID."
    
    """
    url = f"https://german-company-data1.p.rapidapi.com/openregister/ReferenceNumbers.json"
    querystring = {'companyId__exact': companyid_exact, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

