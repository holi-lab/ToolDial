import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def official_atp_players_rankings(timestamp: str, n_players: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve the rankings(**singles**+ **doubles**) of the current tennis season. You can arbitrarily decide the number of players displayed (nplayers) and the time window to refer to (timestamp).
		
		For example, if nplayers = 10,  category= "singles" and timestamp = 2022-04-11 you will receive the top 10 singles standings at the corresponding timestamp (**IMPORTANT**: The timestamp must be in the following format **YYYY-MM-DD** and the date **must fall on Monday** since the rankings are updated at the start of every week)"
    timestamp: Time window

**IMPORTANT**: The timestamp must be in the following format **YYYY-MM-DD** and the date **must fall on Monday** since the rankings are updated at the start of every week
        category: singles or doubles
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/rankings/atp/{category}/{n_players}/{timestamp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_players_rankings(n_player: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint, you can retrieve info about the live tennis rankings for a given number of players, with position/points/info about the last match played in the current active tournament. Please note that in the ATP circuit the official leaderboard is updated every Monday."
    n_player: Max: 500.  Please provide just an Integer
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/live_leaderboard/{n_player}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def atp_player_stats_by_id(year: str, surface: str, player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint you can retrieve official updated statistics for a given ATPplayer. Values are updated after every played match. ID can be retrieved from the Official ATP Players Rankings endpoint."
    year: Specify a year
        surface: Valid options are: clay, grass, hard, all.
        player_id: ID can be retrieved from the Official ATP Players Rankings endpoint.
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/player_stats/atp/{player_id}/{year}/{surface}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_info_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides YTD and Career details for a selected player. You can obtain the Player ID from the WTA Rankings endpoint."
    player_id: Player ID
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/wta/player_info/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_stats_by_id_and_year(player_id: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint you can retrieve official updated statistics for a given WTA player. Values are updated after every played match. ID can be retrieved from the Official WTA Players Rankings endpoint."
    player_id: ID can be retrieved from the Official WTA Players Rankings endpoint.
        year: Just pass the year (e.g. 2022, 2021 ecc..)
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/player_stats/wta/{player_id}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def official_wta_players_rankings(n_players: str, timestamp: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve the rankings(**singles**+ **doubles**) of the current tennis season. You can arbitrarily decide the number of players displayed (nplayers) and the time window to refer to (timestamp).
		
		For example, if nplayers = 10,  category= "singles" and timestamp = 2022-04-11 you will receive the top 10 singles standings at the corresponding timestamp (**IMPORTANT**: The timestamp must be in the following format **YYYY-MM-DD** )"
    timestamp: Time window

**IMPORTANT**: The timestamp must be in the following format **YYYY-MM-DD** 
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/rankings/wta/{category}/{n_players}/{timestamp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_results_by_id_and_year_wta(year: int, tournament_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint, you can retrieve all the matches played in the selected tournament, in a given year. The results are added instantly for every match that ends. You can Retrieve the list of the IDs from the tournament list WTA endpoint."
    year: Year to be selected. Please note that not all the tournaments have been played every year.
        tournament_id: You can Retrieve the list of the IDs from the tournament list WTA endpoint.
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/tournament_results/wta/{tournament_id}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_list_wta(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint, you can retrieve the list of all tournaments to be played in a selected year (from *1960* to *2022*).  We return some basic info about every row such as tournament venue, surface, prizemoney, etc. Results by Id and Year** endpoint"
    year: year must fall between 1960 and 2022
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/tournament_list/wta/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_list(category: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint, you can retrieve the list of all tournaments to be played in a selected year.  We return some basic info about every row such as tournament venue, surface, winners, etc. If you want to have all the details regarding the matches played in a single tournament you can use the **Tournament Results by Id and Year** endpoint"
    category: Options:

- **atpgs**: Atp tournaments + grand Slams

- **atp**: Atp circuit

- **gs**: grand slams

- **1000**: Masters 1000

- **ch**: Challenger Circuit
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/tournament_list/atp/{year}/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_matches_stats(match_id: str, get_10319698: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can retrieve all te details related to a single match such as Aces, Break points, First Serve %, first serve returns, second serves, ecc.."
    match_id: Id identying the single match. It can be retrieved from the Live Scores with Live Betting Odds endpoint.
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/match_details/{match_id}"
    querystring = {}
    if get_10319698:
        querystring['10319698'] = get_10319698
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_scores_with_live_betting_odds(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides info about all the live matches from ATP + WTA. It includes live and prematch odds."
    
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/live_scores"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def global_players_statistics(season: str, category: str, surface: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve global statistics about tennis players. You can customize the statistic (aces, breakpoints converted, 1st serve points won...), the season (all, 2022, 2021...), or the surface(clay, hard, grass).
		You can find the list of available options under every parameter."
    season: Select season to select. Just pass the year:
all- all time rankings

The first year supported is 1991.
        category: Available options are: **aces**, **1stserve**, **returngameswon**, **breakpointsconverted**, **returnpointswon2ndserve**, **returnpointswon2ndserve**, **returnpointswon1stserve**, **1stservepointswon**, **2ndservepointswon**, **servicegameswon**, **breakpointsaved**

        surface: **clay**, **grass**or **hard**. Otherwise, just leave **all**to select all surfaces.
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/global_players_stats/{category}/{season}/{surface}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_results_by_id_and_year(year: int, tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint, you can retrieve all the matches played in the selected tournament, in a given year. The results are added instantly for every match that ends. You can Retrieve the list of the IDs from the tournament_list endpoint."
    year: Year to be selected. Please note that not all the tournaments have been played every year.
        tournament_id: You can Retrieve the list of the IDs from the tournament_list endpoint.
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/tournament_results/{tournament_id}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_ranking_history_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can access all the history of the rankings for a selected player. ID **must** be provided, you can retrieve a player ID from the Players Rankings endpoint."
    player_id: Id of the player. A list of the IDs can be retrieved from the Players Rankings endpoint.
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/ranking_history/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_details_by_id(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve specific details about a single player. ID **must** be provided, you can retrieve a player ID from the Players Rankings endpoint."
    player_id: Id of the player. A list of the IDs can be retrieved from the Players Rankings endpoint.
        
    """
    url = f"https://ultimate-tennis1.p.rapidapi.com/player_info/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-tennis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

