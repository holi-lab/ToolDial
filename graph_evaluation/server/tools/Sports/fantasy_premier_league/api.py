import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_league(name: str, p: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a League by league name. Returns Max 10 results per page"
    name: League Name 
        p: Results Page Number.
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/leagues"
    querystring = {'name': name, }
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_manager(name: str, p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a Manager by real name. Returns Max 10 results per page"
    name: The Name of the Manager You are searching for.
        p: Page Number for results. Note, adding a page number larger than available number of pages will return a 400 error. Page count returned in response
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/managers"
    querystring = {'name': name, }
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def h2h_league(is_id: str, sp: str=None, ep: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return data on a Head-to-Head (H2H) league by provided league id"
    sp: Standings Page. Data has a \\\\\\\"has_next\\\\\\\" boolean to verify if next page exists
        ep: Entries Page. Data has a \\\\\\\"has_next\\\\\\\" boolean to verify if next page exists
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/leagues/h2h/{is_id}"
    querystring = {}
    if sp:
        querystring['sp'] = sp
    if ep:
        querystring['ep'] = ep
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classic_leagues(is_id: str, ep: str=None, sp: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return data on a classic league by provided league id"
    ep: Entries Page. Data has a \"has_next\" boolean to verify if next page exists
        sp: Standings Page. Data has a \"has_next\" boolean to verify if next page exists
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/leagues/classic/{is_id}"
    querystring = {}
    if ep:
        querystring['ep'] = ep
    if sp:
        querystring['sp'] = sp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures(a: str=None, as: str=None, h: str=None, gw: str='10', hs: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of fixtures in the FPL season"
    a: Away Team ID
        as: Away Team Score
        h: Home Team ID
        gw: Gameweek Filter
        hs: Home Team Score
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fixtures"
    querystring = {}
    if a:
        querystring['a'] = a
    if as:
        querystring['as'] = as
    if h:
        querystring['h'] = h
    if gw:
        querystring['gw'] = gw
    if hs:
        querystring['hs'] = hs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_set_piece_notes(team: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of team notes for updates around set-piece takers"
    team: Filter results by the provided team id
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/team/notes/set-pieces"
    querystring = {}
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dream_team(gameweek: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the FPL dream-team for a chosen gameweek"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/team/dream-team/{gameweek}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return data for a chosen team"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/team/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of all teams in the premier league"
    q: Search data by team names, e.g \"Arsenal\"
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/teams"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fpl_player_historic_stats(is_id: str, s: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns FPL player performance stats for previous FPL seasons"
    s: Filter data and return stats only for the provided season, i.e \"2019/20\", \"2014/15\" etc. 
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/history/{is_id}"
    querystring = {}
    if s:
        querystring['s'] = s
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fpl_player_gameweek_stats(is_id: str, gw: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns FPL player performance stats for each gameweek so far in the season"
    gw: Filter data to return stats for only the provided gameweek
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/stats/{is_id}"
    querystring = {}
    if gw:
        querystring['gw'] = gw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fpl_player_chips_used(is_id: str, gw: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the chips used by the provided FPL player, i.e wildcard, triple captain, free hit"
    gw: Filter results by gameweek to find which chips an FPL player used in the provided gameweek
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/chips/{is_id}"
    querystring = {}
    if gw:
        querystring['gw'] = gw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fpl_player_leagues(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of leagues a chosen FPL player is part of"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/player-leagues/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fpl_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the status of FPL after the latest gameweek, i.e. bonus points added, leagues updated"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fpl_player_picks_for_gameweek(is_id: str, gameweek: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return an FPL players selected team for a chosen gameweek"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/fpl/picks/{is_id}/{gameweek}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_history(is_id: str, s: str='2020/21', ast: str=None, gls: str=None, cs: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return FPL statistics for a provided player from previous seasons"
    s: Filter by season, E.g \"2020/21\", \"2019/20\"
        ast: Filter data and return historic season data where the player assisted the provided amount of goals
        gls: Filter data and return historic season data where the player scored the provided amount of goals
        cs: Filter data and return historic season data where the player kept the provided amount of cleansheets
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/player/{is_id}/history"
    querystring = {}
    if s:
        querystring['s'] = s
    if ast:
        querystring['ast'] = ast
    if gls:
        querystring['gls'] = gls
    if cs:
        querystring['cs'] = cs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_results(is_id: str, opp: str=None, hs: str=None, as: str=None, cs: str=None, gls: str=None, ast: str=None, gw: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all match results in the season so far for a player by a provided player id"
    opp: Opponent team id to filter results
        hs: Home Score filter to return data where the home team scored the provided amount of goals
        as: Away Score filter to return data where the away team scored the provided amount of goals
        cs: Filter the data to return the results where the player kept a cleansheet (1 for a cleansheet, 0 for no cleansheet)
        gls: Filter data to return results where the player scored the provided amount of goals
        ast: Filter data to return results where the player assisted the provided amount of goals
        gw: Filter the data to return results that took place on the provided gameweek
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/player/{is_id}/results"
    querystring = {}
    if opp:
        querystring['opp'] = opp
    if hs:
        querystring['hs'] = hs
    if as:
        querystring['as'] = as
    if cs:
        querystring['cs'] = cs
    if gls:
        querystring['gls'] = gls
    if ast:
        querystring['ast'] = ast
    if gw:
        querystring['gw'] = gw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_fixtures(is_id: str, ht: str=None, gw: str=None, at: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all remaining fixtures for a player by a provided player id"
    ht: Home Team ID filter to return fixtures with the provided home team
        gw: gameweek filter to return fixture for the given gameweek
        at: Away Team ID filter to return fixtures with the provided away team
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/player/{is_id}/fixtures"
    querystring = {}
    if ht:
        querystring['ht'] = ht
    if gw:
        querystring['gw'] = gw
    if at:
        querystring['at'] = at
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_simple(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return simplified details of a player by provided player id"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/player/{is_id}/simple"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return details of a player by provide player id"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/player/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_simple(q: str='Jorginho', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of all players included in FPL in a simplified form, only including the player's first name, last name, web name, fpl cost, team id and their fpl photo address"
    q: Search query to filter by player name
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/players/simple"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players(q: str='Jorginho', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of all players included in FPL"
    q: Search query to filter by player name
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/players"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gameweeks_by_month(gw: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of month objects and the corresponding first and final gameweeks that occur in each month"
    gw: Filter by Gameweek
        
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/data/phases"
    querystring = {}
    if gw:
        querystring['gw'] = gw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_positions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return data on each player position"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/data/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def total_fpl_players(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns total number of FPL players for the current season"
    
    """
    url = f"https://fantasy-premier-league3.p.rapidapi.com/data/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-premier-league3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

