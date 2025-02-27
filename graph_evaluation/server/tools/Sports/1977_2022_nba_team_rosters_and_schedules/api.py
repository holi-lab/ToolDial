import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_nba_team_data_from_1977_2022(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grabs all of the 30 NBA Franchise's Data from 1977-2022 into one object."
    
    """
    url = f"https://1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com/elements"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nba_franchise_w_l_schedule_for_a_specific_year(roster_schedule: str, franchise: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grabs NBA Franchise's W/L Schedule for a specific year"
    
    """
    url = f"https://1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com/elements/{franchise}/{year}/{roster_schedule}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nba_franchise_s_specific_year_data(franchise: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grabs NBA Franchise's Roster and Schedule for a specific year"
    
    """
    url = f"https://1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com/elements/{franchise}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nba_franchise_roster_for_a_specific_year(year: int, franchise: str, roster_schedule: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grabs NBA Franchise's Roster for a specific year"
    
    """
    url = f"https://1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com/elements/{franchise}/{year}/{roster_schedule}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_nba_franchise_data(franchise: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grabs all NBA Data for a Franchise from 1977(if applicable)-2022"
    
    """
    url = f"https://1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com/elements/{franchise}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "1977-2022-nba-team-rosters-and-schedules.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

