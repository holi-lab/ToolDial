import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def venue(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns info about a venue"
    
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/venue"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns info about the team"
    
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/team"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_table(group: str, time: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the table for a group"
    group: Group code (A..H)
        time: Let to compute the table considering only the first half (FH) or the second time (SH) results. Default is full time (FT).
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/table"
    querystring = {'group': group, }
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups_and_substitutes_for_a_match(team1: str, team2: str, matchday: int=None, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lineups, substitutes and coaches for a game (only after the game is finished)."
    team1: Home team
        team2: Away team
        matchday: Match day number for the match
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/formations"
    querystring = {'team1': team1, 'team2': team2, }
    if matchday:
        querystring['matchday'] = matchday
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def missing_player_for_a_match(team2: str, team1: str, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all the players who will miss the match and the reason"
    team2: Away team
        team1: Home team
        matchday: Match day number for the match
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/match/missing"
    querystring = {'team2': team2, 'team1': team1, }
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_statistics(team2: str, team1: str, live: bool=None, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns statistics for a match"
    team2: Away team
        team1: Home team
        live: Returns data for a playing match (default false)
        matchday: Match day number for the match
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/match/stats"
    querystring = {'team2': team2, 'team1': team1, }
    if live:
        querystring['live'] = live
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_stats_for_a_match(team2: str, team1: str, player: str, matchday: int=None, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the player's statistics for a match"
    team2: Away team
        team1: Home team
        player: Player name
        matchday: Match day number for the match
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/match/player"
    querystring = {'team2': team2, 'team1': team1, 'player': player, }
    if matchday:
        querystring['matchday'] = matchday
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_events(team1: str, team2: str, matchday: int=None, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the events for a match"
    team1: Home team
        team2: Away team
        matchday: Match day number for the match
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/match/events"
    querystring = {'team1': team1, 'team2': team2, }
    if matchday:
        querystring['matchday'] = matchday
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_scorers(fromdate: str=None, todate: str=None, page: int=None, player: str=None, team: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns top scorers for the Cup or the number of goals scored by a player"
    fromdate: Compute the table only with matches played fromdate (format mmddyyyy)
        todate: Compute the table only with matches played todate (format mmddyyyy)
        page: Page result to return. Default is 1 that are results from 1 to 5 (max. value is 20).
        player: Player name
        team: Team name
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/table/scorers"
    querystring = {}
    if fromdate:
        querystring['fromdate'] = fromdate
    if todate:
        querystring['todate'] = todate
    if page:
        querystring['page'] = page
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_scorers(team1: str, team2: str, live: bool=None, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns when and how the scorers scored in a match"
    team1: Home team
        team2: Away team
        live: Returns data for a playing match (default false)
        matchday: Match day number for the match
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/scorers"
    querystring = {'team1': team1, 'team2': team2, }
    if live:
        querystring['live'] = live
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_odds(team1: str, team2: str, bookmaker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the available odds for a match and a bookmaker"
    team1: Home team
        team2: Away team
        bookmaker: Bokkmaker name
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/listodds"
    querystring = {'team1': team1, 'team2': team2, 'bookmaker': bookmaker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odd_quotas(team2: str, bookmaker: str, odd: str, team1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns quotas for an odd a match and a bookmaker"
    team2: Away team
        bookmaker: Bookmaker name
        odd: Odd name
        team1: Home team
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/odds"
    querystring = {'team2': team2, 'bookmaker': bookmaker, 'odd': odd, 'team1': team1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers_list(team2: str, team1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of the available bookmakers for a match"
    team2: Away team
        team1: Home team
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/listbookmakers"
    querystring = {'team2': team2, 'team1': team1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_results(matchday: int, group: str='A', live: bool=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all matches for a specified match day number"
    matchday: Match day number
        group: Group code (A..K). Mandatory for 1<=matchday<=6
        live: Returns results with live update (default false).
        date: Return matches in the date (format mmddyyyy).
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague"
    querystring = {'matchday': matchday, }
    if group:
        querystring['group'] = group
    if live:
        querystring['live'] = live
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_players(team: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all players for a team. Not available for BASIC plan."
    team: Team name
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/players"
    querystring = {'team': team, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_details(team: str, player: str, honours: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data about a player. Available only with ULTRA and MEGA plans!"
    team: Player team name
        player: Player name
        honours: Returns only the honours for the player
        
    """
    url = f"https://heisenbug-champions-league-live-scores-v1.p.rapidapi.com/api/championsleague/playerdetails"
    querystring = {'team': team, 'player': player, }
    if honours:
        querystring['honours'] = honours
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-champions-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

