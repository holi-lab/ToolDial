import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def player_stats_for_a_match(team2: str, player: str, team1: str, live: bool=None, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the player's statistics for a match."
    team2: Away team
        player: Player name
        team1: Home team
        live: Returns data for a playing match (default false)
        matchday: Match day number
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/match/player"
    querystring = {'team2': team2, 'player': player, 'team1': team1, }
    if live:
        querystring['live'] = live
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_subscribed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all the events subscribed. "
    
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/push/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
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
    group: Group name (ex. A, B and C)
        time: Let to compute the table considering only the first half (FH) or the second time (SH) results. Default is full time (FT).
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/table"
    querystring = {'group': group, }
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_result(matchday: int, live: bool=None, team2: str=None, group: str=None, team1: str=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all matches for a specified match day number."
    matchday: Match day number
        live: Returns results for playing matches 
        team2: Returns all matches with team2 as away team
        group: Searches for matches in group (A,B,C); only for group stage
        team1: Returns all matches with team1 as home team
        date: Returns all matches in date (mmddyyyyy)
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica"
    querystring = {'matchday': matchday, }
    if live:
        querystring['live'] = live
    if team2:
        querystring['team2'] = team2
    if group:
        querystring['group'] = group
    if team1:
        querystring['team1'] = team1
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def missing_players_for_a_match(team2: str, team1: str, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all the players who will miss the match and the reason."
    team2: Away team
        team1: Home team
        matchday: Match day number for the match
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/match/missing"
    querystring = {'team2': team2, 'team1': team1, }
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_scorers(to: int=None, how: str='any', team: str=None, is_from: int=1, player: str=None, fromdate: str=None, mode: str='all', todate: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns top scorers for the Cup or the number of goals scored by a player"
    to: Optional parameter to specify the last match number to consider to compute the result (default last match number played registered on system)
        how: Return only goals scored according the how parameter. Its value can be: left, right, head or any, that is the default.
        team: Team name
        is_from: Optional parameter to specify the first match number to use to compute the result (default 1)
        player: Player name
        fromdate: Compute the table only with matches played fromdate (forma mmddyyyy)
        mode: Optional parameter to restrict the table compute on home or away games (possible values are home, away or all, that is the default)
        todate: Compute the table only with matches played todate (format mmddyyyy)
        page: Page result to return. Default is 1 that are results from 1 to 5 (max. value for page is 20).
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/table/scorers"
    querystring = {}
    if to:
        querystring['to'] = to
    if how:
        querystring['how'] = how
    if team:
        querystring['team'] = team
    if is_from:
        querystring['from'] = is_from
    if player:
        querystring['player'] = player
    if fromdate:
        querystring['fromdate'] = fromdate
    if mode:
        querystring['mode'] = mode
    if todate:
        querystring['todate'] = todate
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups_and_substitutes_for_a_match(team2: str, team1: str, matchday: int=None, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lineups, substitutes and coaches for a game. "
    team2: Away team
        team1: Home team
        matchday: Match day number for the match
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/formations"
    querystring = {'team2': team2, 'team1': team1, }
    if matchday:
        querystring['matchday'] = matchday
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistic_table(stat: str, to: int=None, team: str=None, player: str=None, is_from: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the players table for a specified statistic. "
    stat: Statistic name
        to: Match day number till which compute the table
        team: Team name
        player: Player name
        is_from: Match day number from which compute the table
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/table/playerstat"
    querystring = {'stat': stat, }
    if to:
        querystring['to'] = to
    if team:
        querystring['team'] = team
    if player:
        querystring['player'] = player
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistic_table(stat: str, to: int=None, team: str=None, avg: str='False', mode: str='top', is_from: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Teams table for a specified statistic. "
    stat: Statistic name
        to: Match day number till which compute the table
        team: Team name
        avg: Set to true if you want average statistic value, that is the table computed as (total value statistic/played games). Default false.
        mode: Set with top to get the top 5, with bottom to get the worse 5 teams (default top)
        is_from: Match day number from which to compute the table
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/table/teamstat"
    querystring = {'stat': stat, }
    if to:
        querystring['to'] = to
    if team:
        querystring['team'] = team
    if avg:
        querystring['avg'] = avg
    if mode:
        querystring['mode'] = mode
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cup_players(team: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all the players for a team."
    team: Team name
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/players"
    querystring = {'team': team, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_statistics(team1: str, team2: str, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns statistics for a match."
    team1: Home team
        team2: Away team
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/match/stats"
    querystring = {'team1': team1, 'team2': team2, }
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_events(team2: str, team1: str, matchday: int=None, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the events for a match."
    team2: Away team
        team1: Home team
        matchday: Match day number for the match
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/match/events"
    querystring = {'team2': team2, 'team1': team1, }
    if matchday:
        querystring['matchday'] = matchday
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers_list(team2: str, team1: str, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of the available bookmakers for a match. "
    team2: Away team
        team1: Home team
        matchday: Match day number for the match
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/listbookmakers"
    querystring = {'team2': team2, 'team1': team1, }
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_list(bookmaker: str, team2: str, team1: str, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the available odds for a match and a bookmaker. "
    bookmaker: Bookmaker name
        team2: Away team
        team1: Home team
        matchday: Match day number for the match
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/listodds"
    querystring = {'bookmaker': bookmaker, 'team2': team2, 'team1': team1, }
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odd_quotas(team1: str, odd: str, team2: str, bookmaker: str, matchday: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns quotas for an odd a match and a bookmaker."
    team1: Home team
        odd: Odd name
        team2: Away team
        bookmaker: Bookmaker name
        matchday: Match day number for the match
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/odds"
    querystring = {'team1': team1, 'odd': odd, 'team2': team2, 'bookmaker': bookmaker, }
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_scorers(team2: str, team1: str, live: bool=None, matchday: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns when and how the scorers scored in a match. "
    team2: Away team
        team1: Home team
        live: Returns data for a playing match (default false)
        matchday: Match day number for the match
        
    """
    url = f"https://copa-america.p.rapidapi.com/api/copaamerica/scorers"
    querystring = {'team2': team2, 'team1': team1, }
    if live:
        querystring['live'] = live
    if matchday:
        querystring['matchday'] = matchday
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "copa-america.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

