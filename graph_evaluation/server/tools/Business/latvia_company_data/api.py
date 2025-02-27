import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getaddressfromcompanylinkedid(link_entity_statement_exact: int, labels: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the address of a Latvian comany from the linked ID returned in a previous search"
    
    """
    url = f"https://latvia-company-data.p.rapidapi.com/latvia/entity_addresses.json"
    querystring = {'_link_entity_statement__exact': link_entity_statement_exact, '_labels': labels, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latvia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchpersonbyownerid(link_person_statement_exact: int, labels: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Once an owner id has been returned from a previous search, this API endpoint can be used to search for person details."
    
    """
    url = f"https://latvia-company-data.p.rapidapi.com/latvia/person_names.json"
    querystring = {'_link_person_statement__exact': link_person_statement_exact, '_labels': labels, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latvia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def serchcompanyowneridbycompanyname(name_like: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the owner Identifier(s) of a company by its name, using % as a wild card search"
    
    """
    url = f"https://latvia-company-data.p.rapidapi.com/latvia/person_source_assertedBy"
    querystring = {'name__like': name_like, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latvia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbycompanyid(labels: str, id_exact: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a company by it's Latvian company ID. This will return a linked ID that can be used for further queries."
    
    """
    url = f"https://latvia-company-data.p.rapidapi.com/latvia/entity_identifiers.json"
    querystring = {'_labels': labels, 'id__exact': id_exact, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latvia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbycompanyname(name_like: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Latvian business register by name, using the % symbol as a wildcard"
    
    """
    url = f"https://latvia-company-data.p.rapidapi.com/latvia/entity_statement.json"
    querystring = {'name__like': name_like, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latvia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

