import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_move_jobs(user_id: str, owner_id: str=None, created_at_gt: str=None, created_at_lt: str=None, offset: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    user_id: User ID
        owner_id: filter by owner.
        created_at_gt: filter by created_at greater than the specified ISO date string.
        created_at_lt: filter by created_at lesser than the specified ISO date string.
        offset: an integer count of results to skip.
        limit: an integer number of maximum results to return. Must be less than 201. Defaults to 200.
        
    """
    url = f"https://yippiemove.p.rapidapi.com/v1/users/{user_id}/move_jobs/"
    querystring = {'user-id': user_id, }
    if owner_id:
        querystring['owner__id'] = owner_id
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yippiemove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_users(created_at_gt: str=None, created_at_lt: str=None, offset: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    created_at_gt: filter by created_at greater than the specified ISO date string.
        created_at_lt: filter by created_at lesser than the specified ISO date string.
        offset: an integer count of results to skip.
        limit: an integer number of maximum results to return. Must be less than 201. Defaults to 200.
        
    """
    url = f"https://yippiemove.p.rapidapi.com/v1/users/"
    querystring = {}
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yippiemove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_email_folders_for_a_particular_account(user_id: str, move_job_id: str, source_destination: str, owner_id: str=None, created_at_gt: str=None, created_at_lt: str=None, offset: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    owner_id: filter by owner.
        created_at_gt: filter by created_at greater than the specified ISO date string.
        created_at_lt: filter by created_at lesser than the specified ISO date string.
        offset: an integer count of results to skip.
        limit: an integer number of maximum results to return. Must be less than 201. Defaults to 200.
        
    """
    url = f"https://yippiemove.p.rapidapi.com/v1/users/{user_id}/move_jobs/{move_job_id}/accounts/{source_destination}/email_folders/"
    querystring = {'user-id': user_id, 'move-job-id': move_job_id, 'source-destination': source_destination, }
    if owner_id:
        querystring['owner__id'] = owner_id
    if created_at_gt:
        querystring['created_at__gt'] = created_at_gt
    if created_at_lt:
        querystring['created_at__lt'] = created_at_lt
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yippiemove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_providers(offset: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    offset: An integer count of results to skip.
        limit: An integer number of maximum results to return. Must be less than 201. Defaults to 200.
        
    """
    url = f"https://yippiemove.p.rapidapi.com/v1/providers/"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yippiemove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_specific_users(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    user_id: User ID
        
    """
    url = f"https://yippiemove.p.rapidapi.com/v1/users/{user_id}"
    querystring = {'user-id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yippiemove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_email_part_of_a_job(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://yippiemove.p.rapidapi.com/v1/users/{user_id}/move_jobs/email_part/"
    querystring = {'user-id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yippiemove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_specific_move_jobs(user_id: str, move_job_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    user_id: User ID
        move_job_id: Move Job ID
        
    """
    url = f"https://yippiemove.p.rapidapi.com/v1/users/{user_id}/move_jobs/{move_job_id}/"
    querystring = {'user-id': user_id, 'move-job-id': move_job_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yippiemove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

