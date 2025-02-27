import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def read_one_user(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/users/{is_id}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_changes(schedule_id: str, is_from: str=None, slot: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/changes/{schedule_id}"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if slot:
        querystring['slot'] = slot
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_all_users(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/users.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_appointments(schedule_id: str, limit: int=None, start: str=None, form: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/bookings.json"
    querystring = {'schedule_id': schedule_id, }
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    if form:
        querystring['form'] = form
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available(schedule_id: str, is_from: str=None, length: int=None, resource: str=None, full: bool=None, maxresults: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/free/{schedule_id}"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if length:
        querystring['length'] = length
    if resource:
        querystring['resource'] = resource
    if full:
        querystring['full'] = full
    if maxresults:
        querystring['maxresults'] = maxresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_forms(is_id: str=None, form_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/forms"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if form_id:
        querystring['form_id'] = form_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_agenda(schedule_id: str, user: str=None, is_from: str=None, slot: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/agenda/{schedule_id}"
    querystring = {}
    if user:
        querystring['user'] = user
    if is_from:
        querystring['from'] = is_from
    if slot:
        querystring['slot'] = slot
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_appointments(schedule_id: str, is_from: str=None, slot: bool=None, today: bool=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "it fetches list of appointments between given time range"
    
    """
    url = f"https://supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com/api/range/{schedule_id}"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if slot:
        querystring['slot'] = slot
    if today:
        querystring['today'] = today
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "supersaas-supersaas-online-bookings-and-appointment-scheduling-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

