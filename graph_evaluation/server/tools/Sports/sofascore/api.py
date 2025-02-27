import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def matches_get_player_heatmap(playerid: int, matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player heatmap in a match
		* Link to SVG file to draw the background image of the football pitches https://apidojo.net/heatmap.svg"
    playerid: The value of id field belonging to players returned in .../matches/get-lineups or …/players/search or …/teams/get-squad or …/matches/get-best-players or etc…
        matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-player-heatmap"
    querystring = {'playerId': playerid, 'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_player_statistics(playerid: int, matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics in a match"
    playerid: The value of id field belonging to players returned in .../matches/get-lineups or …/players/search or …/teams/get-squad or …/matches/get-best-players or etc…
        matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-player-statistics"
    querystring = {'playerId': playerid, 'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_esport_games(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get eSport games"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-esport-games"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_media(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest media relating to the  match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-media"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_best_players(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get best players of specific match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-best-players"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_team_streaks(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team streaks information"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-team-streaks"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_statistics(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics of specific match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-statistics"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_graph(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get graph to describe recent form between two teams"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-graph"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_managers(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get managers controlling the match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-managers"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_detail(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match information"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/detail"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_head2head(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head-to-head information between two teams"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-head2head"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_incidents(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get incidents before match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-incidents"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_lineups(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match lineups"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-lineups"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_featured_odds(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured odds relating to a match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-featured-odds"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_all_odds(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds relating to a match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-all-odds"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers_get_next_matches(managerid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent and future matches in which the manager control"
    managerid: The value of id field returned in .../managers/search or .../matches/get-managers or etc ...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/managers/get-next-matches"
    querystring = {'managerId': managerid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers_get_matches_deprecated(managerid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches in which the manager controlled"
    managerid: The value of id field returned in .../managers/search or .../matches/get-managers or etc ...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/managers/get-matches"
    querystring = {'managerId': managerid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers_detail(managerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information of specific manager"
    managerid: The value of id field returned in .../managers/search or .../matches/get-managers or etc ...
        
    """
    url = f"https://sofascore.p.rapidapi.com/managers/detail"
    querystring = {'managerId': managerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers_get_career_history(managerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get career history of specific manager"
    managerid: The value of id field returned in .../managers/search or .../matches/get-managers or etc ...
        
    """
    url = f"https://sofascore.p.rapidapi.com/managers/get-career-history"
    querystring = {'managerId': managerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers_get_last_matches(managerid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last matches in which the manager controlled"
    managerid: The value of id field returned in .../managers/search or .../matches/get-managers or etc ...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/managers/get-last-matches"
    querystring = {'managerId': managerid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers_search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for manager by name"
    name: Get suggested value returned in .../auto-complete endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/managers/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_matches_deprecated(playerid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches in which the player attended"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-matches"
    querystring = {'playerId': playerid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_next_matches(playerid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent matches and future matches in which the player attend"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-next-matches"
    querystring = {'playerId': playerid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_last_matches(playerid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last matches in which the player attended in the past"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-last-matches"
    querystring = {'playerId': playerid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_statistics(tournamentid: int, playerid: int, seasonid: int, type: str='overall', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics of specific player in chosen season and league"
    tournamentid: The id of league returned in .../tournaments/search or .../tournaments/list endpoint
        playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        seasonid: The value of seasons/id JSON object returned in .../teams/get-statistics-seasons endpoint
        type: Get supported values under typesMap JSON object returned in .../teams/get-statistics-seasons endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-statistics"
    querystring = {'tournamentId': tournamentid, 'playerId': playerid, 'seasonId': seasonid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_statistics_seasons(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used together with .../players/get-statistics endpoint to get corresponding seasonId"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-statistics-seasons"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_last_year_summary(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary of specific player in last year"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-last-year-summary"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_transfer_history(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transfer history of specific player"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-transfer-history"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_national_team_statistics(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics of specific player in national team"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-national-team-statistics"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_last_ratings(playerid: int, tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ratings of specific player recently"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        tournamentid: The id of league returned in …/tournaments/search or …/tournaments/list endpoint
        seasonid: The value of seasons/id JSON object returned in …/players/get-statistics-seasons or .../tournaments/get-seasons endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-last-ratings"
    querystring = {'playerId': playerid, 'tournamentId': tournamentid, 'seasonId': seasonid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_attribute_overviews(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get attribute overviews of specific player recently"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-attribute-overviews"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_detail(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information of specific player"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/detail"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_characteristics(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get characteristics of specific player"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-characteristics"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for players by name"
    name: Get suggested value returned in .../auto-complete endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_matches_deprecated(teamid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches in which the team attended"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-matches"
    querystring = {'teamId': teamid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_next_matches(teamid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent matches and future matches in which the team attend"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-next-matches"
    querystring = {'teamId': teamid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_last_matches(teamid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last matches in which the team attended in the past"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-last-matches"
    querystring = {'teamId': teamid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_player_statistics(seasonid: int, tournamentid: int, teamid: int, type: str='overall', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics of specific team in chosen season and league"
    seasonid: The value of seasons/id JSON object returned in .../teams/get-player-statistics-seasons endpoint
        tournamentid: The id of league returned in .../tournaments/search or .../tournaments/list endpoint
        teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        type: Get supported values under typesMap JSON object returned in .../teams/get-player-statistics-seasons endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-player-statistics"
    querystring = {'seasonId': seasonid, 'tournamentId': tournamentid, 'teamId': teamid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_player_statistics_seasons(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used together with .../teams/get-player-statistics endpoint to get corresponding seasonId"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-player-statistics-seasons"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_statistics(seasonid: int, teamid: int, tournamentid: int, type: str='overall', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics of specific team in chosen season and league"
    seasonid: The value of seasons/id JSON object returned in .../teams/get-statistics-seasons endpoint
        teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        tournamentid: The id of league returned in .../tournaments/search or .../tournaments/list endpoint
        type: Get supported values under typesMap JSON object returned in .../teams/get-statistics-seasons endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-statistics"
    querystring = {'seasonId': seasonid, 'teamId': teamid, 'tournamentId': tournamentid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_statistics_seasons(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used together with .../teams/get-statistics endpoint to get corresponding seasonId"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-statistics-seasons"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_near_events(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent events of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-near-events"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_tournaments(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get attended tournaments of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-tournaments"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_rankings(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rankings of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-rankings"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_transfers(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transfers of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-transfers"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_detail(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/detail"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_performance(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get performance of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-performance"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for teams by name"
    name: Get suggested value returned in .../auto-complete endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_logo(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get logo image of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-logo"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_logo(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List logo image of specific league"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-logo"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers_get_image(managerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image of specific manager"
    managerid: The value of id field returned in .../managers/search or .../matches/get-managers or etc ...
        
    """
    url = f"https://sofascore.p.rapidapi.com/managers/get-image"
    querystring = {'managerId': managerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_image(playerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image of specific player"
    playerid: The value of id field returned in .../players/search or .../teams/get-squad or .../matches/get-best-players or etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/players/get-image"
    querystring = {'playerId': playerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_matches_deprecated(seasonid: int, tournamentid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matches of specific league by season"
    seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-matches"
    querystring = {'seasonId': seasonid, 'tournamentId': tournamentid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_next_matches(tournamentid: int, seasonid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current and next matches of specific league by season"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-next-matches"
    querystring = {'tournamentId': tournamentid, 'seasonId': seasonid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_cuptrees(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cuptrees of specific league
		* Some sports support this feature"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-cuptrees"
    querystring = {'tournamentId': tournamentid, 'seasonId': seasonid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_last_matches(tournamentid: int, seasonid: int, pageindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last matches of specific league by season"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        pageindex: For paging purpose
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-last-matches"
    querystring = {'tournamentId': tournamentid, 'seasonId': seasonid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_team_of_the_week_rounds(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used together with .../tournaments/get-team-of-the-week endpoint to get corresponding roundId"
    seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-team-of-the-week-rounds"
    querystring = {'seasonId': seasonid, 'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_media(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest media of specific league"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-media"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_standings(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture of specific league by season"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-standings"
    querystring = {'tournamentId': tournamentid, 'seasonId': seasonid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_rounds(seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List rounds of specific league"
    seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-rounds"
    querystring = {'seasonId': seasonid, 'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_team_of_the_week(roundid: int, seasonid: int, tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team of the week by league, season and round"
    roundid: The value of id field returned in .../tournaments/get-team-of-the-week-rounds endpoint
        seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-team-of-the-week"
    querystring = {'roundId': roundid, 'seasonId': seasonid, 'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_top_players(tournamentid: int, seasonid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top players of specific league"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-top-players"
    querystring = {'tournamentId': tournamentid, 'seasonId': seasonid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_top_teams(tournamentid: int, seasonid: int, type: str='total', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top teams of specific league"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        seasonid: The value of seasons/id JSON object returned in .../tournaments/get-seasons endpoint
        type: One of the following : total|away|home
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-top-teams"
    querystring = {'tournamentId': tournamentid, 'seasonId': seasonid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_seasons(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List seasons of specific league"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-seasons"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_live_events(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List recent live events of specific sport"
    sport: One of the following : football|tennis|basketball|ice-hockey|volleyball|handball|esports|baseball|cricket|motorsport|american-football|rugby|badminton|snooker|darts|futsal|table-tennis|beach-volley|waterpolo|cycling|aussie-rules|floorball|bandy
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-live-events"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_scheduled_events(categoryid: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List scheduled events of specific league"
    categoryid: The value of id field returned in .../categories/list endpoint
        date: The format is yyyy-MM-dd . Ex : 2021-10-26
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-scheduled-events"
    querystring = {'categoryId': categoryid, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for tournaments by name"
    name: Get suggested value returned in .../auto-complete endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_list(categoryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all leagues in specific category or nation"
    categoryid: The value of id field returned in .../tournaments/search or .../categories/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/list"
    querystring = {'categoryId': categoryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_get_featured_events(categoryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List featured events of specific league"
    categoryid: The value of id field returned in .../categories/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/get-featured-events"
    querystring = {'categoryId': categoryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_detail(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List information of specific league"
    tournamentid: The value of id field returned in .../tournaments/search or .../tournaments/list endpoint
        
    """
    url = f"https://sofascore.p.rapidapi.com/tournaments/detail"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list_live(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all categories or nations having live matches currently"
    sport: One of the following : football|tennis|basketball|ice-hockey|volleyball|handball|esports|baseball|cricket|motorsport|american-football|rugby|badminton|snooker|darts|futsal|table-tennis|beach-volley|waterpolo|cycling|aussie-rules|floorball|bandy
        
    """
    url = f"https://sofascore.p.rapidapi.com/categories/list-live"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all categories or nations for tournaments and leagues"
    sport: One of the following : football|tennis|basketball|ice-hockey|volleyball|handball|esports|baseball|cricket|motorsport|american-football|rugby|badminton|snooker|darts|futsal|table-tennis|beach-volley|waterpolo|cycling|aussie-rules|floorball|bandy
        
    """
    url = f"https://sofascore.p.rapidapi.com/categories/list"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions by term or phrase"
    query: Any term or phrase you are familiar with.
        
    """
    url = f"https://sofascore.p.rapidapi.com/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esport_games_get_lineups(gameid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get eSport game's lineups"
    gameid: The id field of match returned in .../matches/get-esport-games
        
    """
    url = f"https://sofascore.p.rapidapi.com/esport-games/get-lineups"
    querystring = {'gameId': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esport_games_get_statistics(gameid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get eSport game's statistics"
    gameid: The id field of match returned in .../matches/get-esport-games
        
    """
    url = f"https://sofascore.p.rapidapi.com/esport-games/get-statistics"
    querystring = {'gameId': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esport_games_get_rounds(gameid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get eSport game's rounds"
    gameid: The id field of match returned in .../matches/get-esport-games
        
    """
    url = f"https://sofascore.p.rapidapi.com/esport-games/get-rounds"
    querystring = {'gameId': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esport_games_get_bans(gameid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get eSport game's bans"
    gameid: The id field of match returned in .../matches/get-esport-games
        
    """
    url = f"https://sofascore.p.rapidapi.com/esport-games/get-bans"
    querystring = {'gameId': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_squad(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get squad of specific team"
    teamid: The value of id field returned in .../teams/search endpoint or .../matches/detail or .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-top-teams, etc...
        
    """
    url = f"https://sofascore.p.rapidapi.com/teams/get-squad"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_votes(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get votes for match, and players"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-votes"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_tweets(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest tweets relating to the  match"
    matchid: The id field of match returned in .../tournaments/get-featured-events or .../tournaments/get-scheduled-events or .../tournaments/get-matches or .../teams/get-matches or .../players/get-matches or .../managers/get-matches
        
    """
    url = f"https://sofascore.p.rapidapi.com/matches/get-tweets"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

