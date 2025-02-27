import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def current_week_scores(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the current weeks scores split by match-up:
		
		- matchupTitle: The name of both teams in the format "{firstOpponentName}  vs. {lastOpponentName}"
		- Scores: An object containing:
		    - firstOpponentName: The name of the first team in the match-up. 
		    - firstOpponentScore: The score (as a float) of the first team in the match-up.
		    - lastOpponentName: The name of the last team in the match-up. 
		    - lastOpponentScore: The score (as a float) of the last team in the match-up.
		
		If the endpoint is called outside of the fantasy leagues season, it will will return the most recent weeks scores."
    
    """
    url = f"https://fantasy-football6.p.rapidapi.com/api/{league_id}/scores"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_teams(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all teams within a given fantasy football league. 
		
		Returns the following information for each team in the given league:
		- teamName - The name of the team.
		- teamId - The unique team ID within the league.
		- teamUrl - The URL suffix to retrieve the team from https://fantasy.nfl.com/"
    
    """
    url = f"https://fantasy-football6.p.rapidapi.com/api/{league_id}/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a health message specifying the port the application is serving on"
    
    """
    url = f"https://fantasy-football6.p.rapidapi.com/_health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-football6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

