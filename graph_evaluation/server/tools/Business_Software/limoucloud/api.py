import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def logout(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Logout from account"
    
    """
    url = f"https://limoucloud.p.rapidapi.com/api/accounts/logout/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "limoucloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_checklists(page_size: int=10, from_date: str='2020-01-02', to_date: str='2020-01-02', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Checklists created by driver"
    
    """
    url = f"https://limoucloud.p.rapidapi.com/api/driver/checklists"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if from_date:
        querystring['from_date'] = from_date
    if to_date:
        querystring['to_date'] = to_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "limoucloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_assigned_vehicle(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Assigned vehicle to add its id in Checklist creation"
    
    """
    url = f"https://limoucloud.p.rapidapi.com/api/driver/get-assigned-vehicle"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "limoucloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reservation_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Reservation by ID"
    
    """
    url = f"https://limoucloud.p.rapidapi.com/api/driver/reservations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "limoucloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reservations(to_date: str='2021-11-20', from_date: str='2021-11-14', page_size: int=10, status: str='CONFIRMED', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all driver related reservations"
    status: Please provide Values from the below list as per your need. 
 'QUOTED '=> Quoted by the client but not confirmed
'CONFIRMED' => Confirmed by Driver (assigned to the driver)
 'SCHEDULED' => Scheduled for later
'COMPLETED'  => Completed trip
'CANCELLED' => Canceled by either party
 'REQUESTED' => Requested to driver (new rides)
        
    """
    url = f"https://limoucloud.p.rapidapi.com/api/driver/reservations"
    querystring = {}
    if to_date:
        querystring['to_date'] = to_date
    if from_date:
        querystring['from_date'] = from_date
    if page_size:
        querystring['page_size'] = page_size
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "limoucloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

