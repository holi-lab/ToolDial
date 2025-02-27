import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tournament_standings(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team Rankings for a specific competition."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://ice-hockey-data.p.rapidapi.com/tournament/standings"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
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
    date: The date of the match. The format is {dd/MM/yyyy}. Match list data can be retrieved for only ± 7 days.
        
    """
    url = f"https://ice-hockey-data.p.rapidapi.com/match/list/results"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
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
    date: The date of the match. The format is {dd/MM/yyyy}. Match list data can be retrieved for only ± 7 days.
        
    """
    url = f"https://ice-hockey-data.p.rapidapi.com/match/list/scheduled"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
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
    url = f"https://ice-hockey-data.p.rapidapi.com/tournament/teams"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
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
    url = f"https://ice-hockey-data.p.rapidapi.com/tournament/info"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
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
    date: The date of the match. The format is {dd/MM/yyyy}. Match list data can be retrieved for only ± 7 days.
        
    """
    url = f"https://ice-hockey-data.p.rapidapi.com/match/list"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_fixture(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full match list with period and final scores."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://ice-hockey-data.p.rapidapi.com/tournament/fixture"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
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
    url = f"https://ice-hockey-data.p.rapidapi.com/tournament/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_live(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including live matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the match. The format is {dd/MM/yyyy}. Match list data can be retrieved for only ± 7 days.
        
    """
    url = f"https://ice-hockey-data.p.rapidapi.com/match/list/live"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_match_list(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the result list of the last 20 matches between the two teams in overall, with home and away filters.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Ice Hockey Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://ice-hockey-data.p.rapidapi.com/h2h/match/list/recent"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ice-hockey-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

