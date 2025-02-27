import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def league_players(team: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all player for a team"
    team: Team name
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/players"
    querystring = {'team': team, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_details(team: str, player: str, honours: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data about a player. Available only for paid subscription!"
    team: Team name
        player: Player name
        honours: Return only the honours for the player (default false)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/playerdetails"
    querystring = {'team': team, 'player': player, }
    if honours:
        querystring['honours'] = honours
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_table(mode: str=None, time: str=None, is_from: int=None, to: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the table for the league"
    mode: Optional parameter to restrict the table compute on home or away games (default all)
        time: Let to compute the table considering only the first half (FH) or the second time (SH) results. Default is full time (FT).
        is_from: Optional parameter to specify the first match number to use to compute the table (default 1)
        to: Optional parameter to specify the last match number to consider to compute the table (default last match number played registered on system)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/table"
    querystring = {}
    if mode:
        querystring['mode'] = mode
    if time:
        querystring['time'] = time
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_statistics(team1: str, team2: str, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns match statistics"
    team1: Home team
        team2: Away team
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/match/stats"
    querystring = {'team1': team1, 'team2': team2, }
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def missing_players_for_a_match(team1: str, team2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns missing players for a match"
    team1: Home team
        team2: Away team
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/match/missing"
    querystring = {'team1': team1, 'team2': team2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
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
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/listbookmakers"
    querystring = {'team2': team2, 'team1': team1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_list(team1: str, team2: str, bookmaker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the available odds for a match and a bookmaker"
    team1: Home team
        team2: Away team
        bookmaker: Bookmaker name
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/listodds"
    querystring = {'team1': team1, 'team2': team2, 'bookmaker': bookmaker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_stats_for_a_match(player: str, team1: str, team2: str, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the player's statistics for a match"
    player: Player name
        team1: Home team
        team2: Away team
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/match/player"
    querystring = {'player': player, 'team1': team1, 'team2': team2, }
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_matches_results(matchday: int, team1: str=None, team2: str=None, live: bool=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns season matches results"
    matchday: Match day for games
        team1: Returns all matches with team1
        team2: Return all matches with team2
        live: Returns only playing matches with live updates (default false)
        date: Returns all matches for the date (format mmddyyyy)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague"
    querystring = {'matchday': matchday, }
    if team1:
        querystring['team1'] = team1
    if team2:
        querystring['team2'] = team2
    if live:
        querystring['live'] = live
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_scorers(how: str=None, page: int=None, mode: str=None, player: str=None, team: str=None, is_from: str=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns top scorers for the league"
    how: Return only goals scored according the how parameter. Its value can be: left, right, head or any, that is the default.
        page: Page result to return. Default is 1 that are results from 1 to 5 (max. value is 20).
        mode: Optional parameter to restrict the table compute on home or away games (default all)
        player: Optional parameter to get goals number only for the player name specified
        team: Team name
        is_from: Optional parameter to specify the first match number to use to compute the result (default 1)
        to: Optional parameter to specify the last match number to consider to compute the result (default last match number played registered on system)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/table/scorers"
    querystring = {}
    if how:
        querystring['how'] = how
    if page:
        querystring['page'] = page
    if mode:
        querystring['mode'] = mode
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odd_quotas(bookmaker: str, odd: str, team1: str, team2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns quotas for an odd a match and a bookmaker"
    bookmaker: Bookmaker name
        odd: Odd name
        team1: Home team
        team2: Away team
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/odds"
    querystring = {'bookmaker': bookmaker, 'odd': odd, 'team1': team1, 'team2': team2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_scorers(team1: str, team2: str, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns who scored and how for a match"
    team1: Home team
        team2: Away team
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/scorers"
    querystring = {'team1': team1, 'team2': team2, }
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_events(team1: str, team2: str, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the events for a match (yellow and red cards, substitutions, shots on post and formations module)"
    team1: Home team
        team2: Away team
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/match/events"
    querystring = {'team1': team1, 'team2': team2, }
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups_and_substitutes_for_a_game(team1: str, team2: str, live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lineups, substitutes and coaches for a game after a while the game is finished."
    team1: Home team
        team2: Away team
        live: Returns data for a playing match (default false)
        
    """
    url = f"https://heisenbug-russian-league-live-scores-v1.p.rapidapi.com/api/russianleague/formations"
    querystring = {'team1': team1, 'team2': team2, }
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-russian-league-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

