import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_metadata(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the metadata associated with a contract in your contract database."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/metadata"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pdf(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests the PDF version of a contract using the contract's unique identifier."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/pdf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_versions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all versions of a contract associated with a unique identifier."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_a_new_version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates a new, updated version of a contract using a unique identifier (uid) to identify the contract."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_a_new_pdf_version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates a new version of a contract (associated with a unique identifier) in PDF format."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/version/upload"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the latest version of a contract based upon the unique identifier passed as a parameter."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/version/last"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_members(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of users who have access to a contract stored in Concord's contract management system. A unique contract identifier is passed as a parameter."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/member"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def edit_a_member(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows for editing user permissions for users who have access to contracts stored on Concord's contract management system."
    
    """
    url = f"https://concordworldwide-concord-contract-management-v1.p.rapidapi.com/api/rest/1/agreement/uid/member"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concordworldwide-concord-contract-management-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

