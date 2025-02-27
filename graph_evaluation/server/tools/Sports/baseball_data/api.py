import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def daily_match_list_live(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including live matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the matches. The format is {dd/MM/yyyy}. Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/match/list/live"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of tournaments in your data coverage."
    
    """
    url = f"https://baseball-data.p.rapidapi.com/tournament/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_scheduled(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including scheduled matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the matches. The format is {dd/MM/yyyy}. Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/match/list/scheduled"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_standings(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team Rankings for a specific competition."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/tournament/standings"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_results(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including finished matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the matches. The format is {dd/MM/yyyy}. Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/match/list/results"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_all(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including scheduled, live and finished matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the matches. The format is {dd/MM/yyyy}.  Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/match/list"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_fixture(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full match list with inning and final scores."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/tournament/fixture"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_info(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current season, stage structure(divisions,conferences etc.), country and many more information about a tournament."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/tournament/info"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_teams(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of teams participating in a specific tournament."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://baseball-data.p.rapidapi.com/tournament/teams"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

