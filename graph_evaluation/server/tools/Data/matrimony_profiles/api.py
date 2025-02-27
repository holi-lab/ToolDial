import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_profile_details(member_id: str, method: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Profile Details of Member for given member_id"
    member_id: Value member_id can be found in Get Profiles api result.
        
    """
    url = f"https://matrimony-profiles.p.rapidapi.com/matrimony/api.php"
    querystring = {'member_id': member_id, 'method': method, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "matrimony-profiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_castes(caste_religion: str, method: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Castes"
    caste_religion: this is compulsory parameter, which uses one of the values returned on Get Religions api
        
    """
    url = f"https://matrimony-profiles.p.rapidapi.com/matrimony/api.php"
    querystring = {'caste_religion': caste_religion, 'method': method, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "matrimony-profiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_religions(method: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Religions"
    
    """
    url = f"https://matrimony-profiles.p.rapidapi.com/matrimony/api.php"
    querystring = {'method': method, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "matrimony-profiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profiles(method: str, member_pref_marital_status: str='Never married', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 200 profiles"
    member_pref_marital_status: member_pref_marital_status is an optional parameter.
You can leave it blank or can use any combination of the following values.
Never married
Widowed
Divorced
Awaiting divorce

        
    """
    url = f"https://matrimony-profiles.p.rapidapi.com/matrimony/api.php"
    querystring = {'method': method, }
    if member_pref_marital_status:
        querystring['member_pref_marital_status'] = member_pref_marital_status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "matrimony-profiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

