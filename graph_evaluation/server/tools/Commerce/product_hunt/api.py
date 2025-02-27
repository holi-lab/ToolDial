import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_users(per_page: str, older: str=None, newer: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all users"
    per_page: Filter parameter: define the amount of records sent per call (max 100)
        older: Filter parameter: get only records older than the provided id
        newer: Filter parameter: get only records newer than the provided id
        
    """
    url = f"https://vladtheshark-Product-Hunt-v1.p.rapidapi.com/v1/users"
    querystring = {'per_page': per_page, }
    if older:
        querystring['older'] = older
    if newer:
        querystring['newer'] = newer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vladtheshark-Product-Hunt-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_votes(post_id: str, older: str=None, newer: str=None, per_page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all votes `newer` than the provided id"
    post_id: The ID of the post you want to find votes for
        older: Filter parameter: get only records older than the provided id
        newer: Filter parameter: get only records newer than the provided id
        per_page: Filter parameter: define the amount of records sent per call (max 100)
        
    """
    url = f"https://vladtheshark-Product-Hunt-v1.p.rapidapi.com/v1/posts/1/votes?newer=3"
    querystring = {'post_id': post_id, }
    if older:
        querystring['older'] = older
    if newer:
        querystring['newer'] = newer
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vladtheshark-Product-Hunt-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_day(days_ago: str, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "request a specific day with the `day` parameter"
    days_ago: Parameter for pagination
        day: Alternate parameter for requesting specific days (Format: day=YYYYY-MM-DD
        
    """
    url = f"https://vladtheshark-Product-Hunt-v1.p.rapidapi.com/v1/posts?day=2014-08-10"
    querystring = {'days_ago': days_ago, 'day': day, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vladtheshark-Product-Hunt-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_votes(user_id: str, per_page: int=100, newer: str=None, older: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See all votes for a post"
    user_id: The ID of the user you want to find votes for
        per_page: Filter parameter: define the amount of records sent per call (max 100)
        newer: Filter parameter: get only records newer than the provided id
        older: Filter parameter: get only records older than the provided id
        
    """
    url = f"https://vladtheshark-Product-Hunt-v1.p.rapidapi.com/v1/users/7/votes"
    querystring = {'user_id': user_id, }
    if per_page:
        querystring['per_page'] = per_page
    if newer:
        querystring['newer'] = newer
    if older:
        querystring['older'] = older
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vladtheshark-Product-Hunt-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details_of_a_post(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a post"
    id: The numeric ID of the Post you want to fetch
        
    """
    url = f"https://vladtheshark-Product-Hunt-v1.p.rapidapi.com/v1/posts/1"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vladtheshark-Product-Hunt-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def your_own_details(email: str='ryan@producthunt.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get your own details"
    email: Your user's email
        
    """
    url = f"https://vladtheshark-Product-Hunt-v1.p.rapidapi.com/v1/me"
    querystring = {}
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vladtheshark-Product-Hunt-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details_of_a_user(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a user"
    id: The ID or username of the Post you want to fetch
        
    """
    url = f"https://vladtheshark-Product-Hunt-v1.p.rapidapi.com/v1/users/producthunter225"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vladtheshark-Product-Hunt-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

