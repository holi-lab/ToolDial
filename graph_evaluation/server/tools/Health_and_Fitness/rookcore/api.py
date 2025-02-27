import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retriveuserinformation(user_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/users/{user_uuid}/information"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retriveserverdatetime(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/server/datetime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrivesummariestypes(page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/summaries"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retriveclientusers(search: str='ein', from_date: str='2021-11-12', page: int=1, to_date: str='2021-11-12', search_column: str='email', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    search: Optional, data for search
        from_date: Optional, initial end of period
        to_date: Optional, end date of period
        search_column: Optional, column for search
        
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/users"
    querystring = {}
    if search:
        querystring['search'] = search
    if from_date:
        querystring['from_date'] = from_date
    if page:
        querystring['page'] = page
    if to_date:
        querystring['to_date'] = to_date
    if search_column:
        querystring['search_column'] = search_column
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrivetrainingtypes(search_column: str='step_options', from_date: str='2021-09-14', to_date: str='2021-09-14', search: str='jumps', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    search_column: Optional, column for search
        from_date: Optional, initial end of period
        to_date: Optional, end date of period
        search: Optional, data for search
        
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/trainings/type"
    querystring = {}
    if search_column:
        querystring['search_column'] = search_column
    if from_date:
        querystring['from_date'] = from_date
    if to_date:
        querystring['to_date'] = to_date
    if search:
        querystring['search'] = search
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveuserstatus(user_email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/users/{user_email}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrivesensorsfromuser(user_uuid: str, ownership_type: str='owned', to_date: str='2021-02-22', from_date: str='2021-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    to_date: Optional, end date of period
        from_date: Optional, initial end of period
        
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/sensors/user/{user_uuid}"
    querystring = {}
    if ownership_type:
        querystring['ownership_type'] = ownership_type
    if to_date:
        querystring['to_date'] = to_date
    if from_date:
        querystring['from_date'] = from_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveuserphysiologicalvariables(user_uuid: str, to_date: str='2021-02-22', from_date: str='2021-02-19', type_variable: str='weight', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    to_date: Optional, end date of period
        from_date: Optional, initial end of period
        
    """
    url = f"https://rookcore.p.rapidapi.com/api/v2/users/{user_uuid}/variables"
    querystring = {}
    if to_date:
        querystring['to_date'] = to_date
    if from_date:
        querystring['from_date'] = from_date
    if type_variable:
        querystring['type_variable'] = type_variable
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rookcore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

