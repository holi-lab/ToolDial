import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_teams(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve list of teams for which there is data available."
    
    """
    url = f"https://sport-outcomes.p.rapidapi.com/AllTeams"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Leagues currently covered within the Data services"
    sport: 1 = Soccer
2 = Hockey
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/Leagues"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def koi_glossary(sport: str, languagecode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive a glossary for all KOI's currently offered"
    sport: 1 = Soccer
2 = Hockey
        languagecode: en = English
ru = Russian
fr = French 
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/KOI-Glossary"
    querystring = {'sport': sport, 'languagecode': languagecode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scoreclassifications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Score Classifications as defined in the KOIs"
    
    """
    url = f"https://sport-outcomes.p.rapidapi.com/ScoreClassifications"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goal_s_by_classification(classification: int, teamid: int, sport: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive Team Goals scored based on classification requested in structured JSON (English)"
    classification: The classification (See EndPoint => /GoalClassifications)
        teamid: The Team ID (See EndPoint => /AllTeams)
        sport: 1 = Soccer
2 = Hockey
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/GoalsByClassification"
    querystring = {'classification': classification, 'TeamId': teamid, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_s_fixtures(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Todays games and the current status of them."
    sport: 1 = Soccer
2 = Hockey
3= Basketball
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/TodaysFixtures"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def koi_numerics(gameminute: int, sport: int, matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive KOI's in raw numeric form"
    gameminute: The time elapsed in event, supply zero (0) to get pre match KOI's
        sport: The event Sport, 1= Soccer , 2 = Hockey , 3= Basketball
        matchid: The Event ID (See EndPoint => /TodaysFixtures)
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/KOI-Numerics"
    querystring = {'gameMinute': gameminute, 'sport': sport, 'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kois(gameminute: int, matchid: int, sport: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive KOI's narratives (English)"
    gameminute: The time elapsed in event, supply zero (0) to get pre match KOI's
        matchid: The Event ID (See EndPoint => /TodaysFixtures)
        sport: The event Sport, 1= Soccer , 2 = Hockey , 3 = Basketball
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/KOIs"
    querystring = {'gameMinute': gameminute, 'matchId': matchid, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goal_s_by_fixture_raw(sport: int, teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive Goals scored with classification in raw string format (English)"
    sport: 1 = Soccer
2 = Hockey
3 = Basketball
        teamid: The Team ID (See EndPoint => /AllTeams)
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/GoalsRaw"
    querystring = {'sport': sport, 'TeamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goal_s_by_fixture(teamid: int, sport: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Goals scored with classification in structured JSON (English)."
    teamid: The Team ID (See EndPoint => /AllTeams)
        sport: 1 = Soccer
2 = Hockey
3 = Basketball
        
    """
    url = f"https://sport-outcomes.p.rapidapi.com/GoalsByFixture"
    querystring = {'TeamId': teamid, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-outcomes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

