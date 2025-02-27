import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getaddressfromcompanylinkedid(link_entity_statement_exact: int, labels: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the company address using the linked ID found in previous calls."
    
    """
    url = f"https://slovakia-company-data.p.rapidapi.com/slovakia/entity_addresses.json"
    querystring = {'_link_entity_statement__exact': link_entity_statement_exact, '_labels': labels, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "slovakia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbycompanyname(name_like: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Slovak company register using the company name. Using % as a wildcard."
    
    """
    url = f"https://slovakia-company-data.p.rapidapi.com/slovakia/entity_statement.json"
    querystring = {'name__like': name_like, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "slovakia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbycompanyid(id_exact: int, labels: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Seach the slovak business register using the company ID."
    
    """
    url = f"https://slovakia-company-data.p.rapidapi.com/slovakia/entity_identifiers.json"
    querystring = {'id__exact': id_exact, '_labels': labels, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "slovakia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchpersonbyownerid(labels: str, link_person_statement_exact: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by a person (business owner) from the Slovak business register using the linked ID."
    
    """
    url = f"https://slovakia-company-data.p.rapidapi.com/slovakia/person_names.json"
    querystring = {'_labels': labels, '_link_person_statement__exact': link_person_statement_exact, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "slovakia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

