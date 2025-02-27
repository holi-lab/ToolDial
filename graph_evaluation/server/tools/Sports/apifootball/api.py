import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def players(action: str='get_players', player_name: str='Ronaldo Cristiano', player_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns available players"
    player_name: Player Name - mandatory if player id is not set
        player_id: Player ID - mandatory if player name is not set
        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if action:
        querystring['action'] = action
    if player_name:
        querystring['player_name'] = player_name
    if player_id:
        querystring['player_id'] = player_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(league_id: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns standings for leagues included in your current subscription plan"
    league_id: League internal code
        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {'league_id': league_id, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(action: str='get_countries', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of supported countries included in your current subscription plan"
    
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if action:
        querystring['action'] = action
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions(action: str='get_leagues', country_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of supported competitions included in your current subscription plan"
    country_id: Country ID - if set only leagues from specific country will be returned (Optional)
        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if action:
        querystring['action'] = action
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(action: str, team_id: str=None, league_id: str='148', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of available teams"
    team_id: Team ID - team id mandatory if league id is not set
        league_id: League ID - league id mandatory if team id is not set
        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {'action': action, }
    if team_id:
        querystring['team_id'] = team_id
    if league_id:
        querystring['league_id'] = league_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_results_fixtures(action: str, country_id: str=None, league_id: str=None, match_id: str=None, team_id: str=None, is_from: str='2020-01-23', timezone: str=None, to: str='2020-01-23', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns events included in your current subscription plan"
    country_id: Country ID - if set only leagues from specific country will be returned (Optional)
        league_id: League ID - if set events from specific league will be returned (Optional)
        match_id: Match ID - if set only details from specific match will be returned (Optional)
        team_id: Team ID - if set only details from specific team will be returned (Optional)
        is_from: Start date (yyyy-mm-dd)
        timezone: Default timezone: Europe/Berlin. With this filter you can set the timezone where you want to receive the data. Timezone is in TZ format (exemple: America/New_York)
        to: Stop date (yyyy-mm-dd)
        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {'action': action, }
    if country_id:
        querystring['country_id'] = country_id
    if league_id:
        querystring['league_id'] = league_id
    if match_id:
        querystring['match_id'] = match_id
    if team_id:
        querystring['team_id'] = team_id
    if is_from:
        querystring['from'] = is_from
    if timezone:
        querystring['timezone'] = timezone
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups(action: str='get_lineups', match_id: str='24562', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lineups of one event"
    
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if action:
        querystring['action'] = action
    if match_id:
        querystring['match_id'] = match_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics(match_id: str='24562', action: str='get_statistics', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns statistics of one event"
    
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if match_id:
        querystring['match_id'] = match_id
    if action:
        querystring['action'] = action
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(match_id: str='194200', action: str='get_odds', to: str=None, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns odds (1x2, BTS, O/U, AH) for events included in your current subscription plan"
    match_id: Match ID - if set only odds from specific event will be returned (Optional)
        to: Stop date (yyyy-mm-dd)
        is_from: Start date (yyyy-mm-dd)
        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if match_id:
        querystring['match_id'] = match_id
    if action:
        querystring['action'] = action
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def h2h(secondteam: str='Arsenal', firstteam: str='Chelsea', action: str='get_H2H', timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the last games between submiteted teams and the last games of each team"
    secondteam: Second team name
        firstteam: First team name
        timezone: Default timezone: Europe/Berlin. With this filter you can set the timezone where you want to receive the data. Timezone is in TZ format (exemple: America/New_York). (Optional)

        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if secondteam:
        querystring['secondTeam'] = secondteam
    if firstteam:
        querystring['firstTeam'] = firstteam
    if action:
        querystring['action'] = action
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predictions(to: str='2020-01-23', action: str='get_predictions', country_id: str=None, match_id: str=None, league_id: str=None, is_from: str='2020-01-23', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns mathematical calculated predictions for the events included in your current subscription plan"
    to: Stop date (yyyy-mm-dd)
        country_id: Country ID - if set only leagues from specific country will be returned (Optional)
        match_id: Match ID - if set only details from specific match will be returned (Optional)
        league_id: League ID - if set events from specific league will be returned (Optional)
        is_from: Start date (yyyy-mm-dd)
        
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if to:
        querystring['to'] = to
    if action:
        querystring['action'] = action
    if country_id:
        querystring['country_id'] = country_id
    if match_id:
        querystring['match_id'] = match_id
    if league_id:
        querystring['league_id'] = league_id
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livescore(match_live: str='1', action: str='get_events', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns events live included in your current subscription plan"
    
    """
    url = f"https://apifootball3.p.rapidapi.com/"
    querystring = {}
    if match_live:
        querystring['match_live'] = match_live
    if action:
        querystring['action'] = action
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apifootball3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

