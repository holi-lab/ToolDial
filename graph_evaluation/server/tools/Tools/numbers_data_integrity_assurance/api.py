import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def read_asset_from_workflow_id(workflow_id: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read asset information such as creator wallet, NFT tokens, creation proof, etc, viaworkflow ID"
    workflow_id: Workflow ID returned during asset creation.
        token: User token
        
    """
    url = f"https://numbers-data-integrity-assurance.p.rapidapi.com/read_asset_from_workflow"
    querystring = {'workflow_id': workflow_id, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers-data-integrity-assurance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_asset(nid: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read asset information such as creator wallet, NFT tokens, creation proof, etc."
    nid: Numbers ID of the asset
        token: User token
        
    """
    url = f"https://numbers-data-integrity-assurance.p.rapidapi.com/read_asset"
    querystring = {'nid': nid, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers-data-integrity-assurance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_information(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read registered user information."
    token: Token returned after login.
        
    """
    url = f"https://numbers-data-integrity-assurance.p.rapidapi.com/me"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers-data-integrity-assurance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

