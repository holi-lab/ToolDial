import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_roster_information(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the team roster information for a especific NBA team."
    teamid: Team id
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbateamplayers"
    querystring = {'teamid': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nba_team_info(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the team info for a specific NBA team."
    teamid: Team id
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbateaminfo"
    querystring = {'teamid': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nba_team_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the list of all NBA teams their identification info for ESPN."
    
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbateamlist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nba_teams_standings(year: str, group: str='conference', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the team standings for the NBA."
    year: Year   (YYYY)
        group: Group Names: league, conference, division
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbastandings"
    querystring = {'year': year, }
    if group:
        querystring['group'] = group
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nba_scoreboard(day: str, year: str, month: str, limit: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the NBA scoreboard data for a specified date if available.
		
		   /*
		       Param	Type	Description
		    year     	 *   	Year   (YYYY)
		    mont         *   	Month (MM)
		     day         *	    Day (DD)
		
		    */"
    day: Day (DD)
        year: Year   (YYYY)
        month: Month (MM)
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbascoreboard"
    querystring = {'day': day, 'year': year, 'month': month, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nba_shedules(year: str, month: str, day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the NBA schedule data for a specified date if available.
		
		
		  /*
		       Param	Type	Description
		    year     	 *   	Year   (YYYY)
		    mont         *   	Month (MM)
		     day         *	    Day (DD)
		
		    */"
    year: Year   (YYYY)
        month: Month (MM)
        day: Day (DD)
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbaschedule"
    querystring = {'year': year, 'month': month, 'day': day, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_pickcenter(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the NBA game PickCenter data for a specified game."
    id: Game id
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbapick"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_summary(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the NBA game summary data for a specified game."
    id: Game id
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbasummary"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_box_score(is_id: str='401283399', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the NBA game box score data for a specified game.
		
		Parameters_>
		id - Game id"
    is_id: Game id
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbabox"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_play_by_play(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Gets the NBA game play-by-play data."
    id: Game id
        
    """
    url = f"https://api-basketball-nba.p.rapidapi.com/nbaplay"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

