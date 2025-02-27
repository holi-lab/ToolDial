import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_2d3d_calendar_database(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint where you can retrieve all the previous information of 2D Results and 3D Results with outputting 30 different results."
    
    """
    url = f"https://myanmar-all-in-one-2d-results.p.rapidapi.com/api/v1/calendar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myanmar-all-in-one-2d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2d3d_daily_results(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will help you out where you can retrieve all the data for 12:01PM results and 04:31PM results as well as 3D lottery."
    
    """
    url = f"https://myanmar-all-in-one-2d-results.p.rapidapi.com/api/v1/daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myanmar-all-in-one-2d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def myanmar_local_private_exchange_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint will help you out to retrieve some of the local private bank daily currency exchange rates which is AYA, AGD, YOMA, MCB, UAB, CB and much more..."
    
    """
    url = f"https://myanmar-all-in-one-2d-results.p.rapidapi.com/api/v1/currency"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myanmar-all-in-one-2d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2d_calendar_history_free_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To retrieve all the 2D Live Histories since 2019 until now."
    
    """
    url = f"https://myanmar-all-in-one-2d-results.p.rapidapi.com/api/v1/chen-calendar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myanmar-all-in-one-2d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holiday(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can retrieve all  entire year holiday result alongside with Thai Stock Changed Market."
    
    """
    url = f"https://myanmar-all-in-one-2d-results.p.rapidapi.com/api/v1/holiday"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myanmar-all-in-one-2d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def realtime_result(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this endpoint you can retrieve the realtime result data alongside with simultaneously updating Thailand Stock Exchanged data."
    
    """
    url = f"https://myanmar-all-in-one-2d-results.p.rapidapi.com/api/v1/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myanmar-all-in-one-2d-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

