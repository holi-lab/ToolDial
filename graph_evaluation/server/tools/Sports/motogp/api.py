import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_events_from_a_season_id_events_mean_all_completed_and_pending_races(season_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events from a season id [Events mean all completed and pending races]"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_all_events_from_season"
    querystring = {'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_riders_of_a_season_requires_rider_category_and_season_year(season_year: int, category_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all riders of a season [Requires Rider category and season year]"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_all_riders_of_season"
    querystring = {'season_year': season_year, 'category_id': category_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_rider_categories_for_a_season(season_year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all rider categories for a season"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_all_rider_categories"
    querystring = {'season_year': season_year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rider_details_by_id(rider_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rider details by id"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_rider_info_by_id"
    querystring = {'rider_id': rider_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_summary_of_a_rider_by_id(rider_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary of a rider by id"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_rider_summary_from_rider_id"
    querystring = {'rider_id': rider_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_statistics_of_a_rider(rider_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics of a rider"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_rider_statistics_from_rider_id"
    querystring = {'rider_id': rider_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rider_information_by_name(first_name: str, last_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rider information by name"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_rider_info_by_name"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_race_result_from_a_race_session_race_session_id_is_required(session_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get race result from a race session [Race session id is required]"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_race_result_from_session"
    querystring = {'session_id': session_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_categories_from_an_event_category_is_required_to_get_race_data(event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all categories from an event [Category is required to get race data]"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_categories_from_event_id"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_seasons_required_for_race_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all seasons [Required for Race Data] - This gives you a season id from which you can get all events"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_all_seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_race_sessions_from_category_id_and_event_id_race_sessions_race_sprint_race_fp1_fp2(category_id: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get race sessions from category id and event id [Race sessions - Race/Sprint Race/FP1/FP2...]"
    
    """
    url = f"https://motogp2.p.rapidapi.com/get_race_sessions_from_category_and_event"
    querystring = {'category_id': category_id, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_available_seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available seasons"
    
    """
    url = f"https://motogp2.p.rapidapi.com/api/data/get_all_seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

