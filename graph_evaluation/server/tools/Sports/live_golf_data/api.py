import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def world_ranking(year: str, statid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the world rankings for a given year"
    
    """
    url = f"https://live-golf-data.p.rapidapi.com/stats"
    querystring = {'year': year, 'statId': statid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings(year: str, tournid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the player earnings for a given `tournId` and `year`."
    
    """
    url = f"https://live-golf-data.p.rapidapi.com/earnings"
    querystring = {'year': year, 'tournId': tournid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def points(tournid: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch official FedExCup points earned per player for a given `tournId` and `year`. Note that PGA Tour non-members will not be officially ranked."
    
    """
    url = f"https://live-golf-data.p.rapidapi.com/points"
    querystring = {'tournId': tournid, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments(orgid: str, tournid: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch tournament information, such as metadata, players, courses, and earnings. Query the `schedule` endpoint to retrieve the necessary `tournId` and `year` parameters. This info will be updated with a players entry list the Friday before the tournament."
    orgid: Choose 1 for PGA Tour or  2 for LIV Tour.
        
    """
    url = f"https://live-golf-data.p.rapidapi.com/tournament"
    querystring = {'orgId': orgid, 'tournId': tournid, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players(lastname: str='Morikawa', playerid: str='50525', firstname: str='Collin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a PGA Tour player by his last name, first name, playerId, or a combination of these parameters.
		
		You must provide at least one search parameter to the API."
    
    """
    url = f"https://live-golf-data.p.rapidapi.com/players"
    querystring = {}
    if lastname:
        querystring['lastName'] = lastname
    if playerid:
        querystring['playerId'] = playerid
    if firstname:
        querystring['firstName'] = firstname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedules(orgid: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the schedule for the desired year and organization."
    orgid: Choose 1 for PGA Tour or  2 for LIV Tour.
        
    """
    url = f"https://live-golf-data.p.rapidapi.com/schedule"
    querystring = {'orgId': orgid, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def organizations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the organizations that are supported by this golf data API."
    
    """
    url = f"https://live-golf-data.p.rapidapi.com/organizations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fedexcup_ranking(year: str, statid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the FedExCup ranking leaderboard for a given year. These rankings are updated live during a tournament to show the player's projected ranking."
    
    """
    url = f"https://live-golf-data.p.rapidapi.com/stats"
    querystring = {'year': year, 'statId': statid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scorecards(orgid: str, year: str, playerid: str, tournid: str, roundid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a scorecard for a given `tournId`, `year`, `playerId`, and `roundId`.  Scorecards include shot-by-shot granularity."
    orgid: Choose 1 for PGA Tour or  2 for LIV Tour.
        
    """
    url = f"https://live-golf-data.p.rapidapi.com/scorecard"
    querystring = {'orgId': orgid, 'year': year, 'playerId': playerid, 'tournId': tournid, }
    if roundid:
        querystring['roundId'] = roundid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaderboards(orgid: str, tournid: str, year: str, roundid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the most recent leaderboard for a given `tournId`, `year`,  and `roundId`. Query the `schedule` endpoint for the desired `tournId` and `year`.
		
		Leaderboards are available 1-2 days before the event begins."
    orgid: Choose 1 for PGA Tour or  2 for LIV Tour.
        
    """
    url = f"https://live-golf-data.p.rapidapi.com/leaderboard"
    querystring = {'orgId': orgid, 'tournId': tournid, 'year': year, }
    if roundid:
        querystring['roundId'] = roundid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-golf-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

