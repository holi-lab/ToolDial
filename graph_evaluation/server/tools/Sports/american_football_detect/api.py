import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def standings(league_id: str='eq.828', limit: str='50', type: str='eq.total', season_id: str='eq.757', is_id: str='eq.68147', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all standings.
		For the type argument you can search by: **home**, **away** and **total**.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated after every finished match.<br />**Recommended Calls**: 1 call per finished match.
		
		### Use Cases
		Get all standings<br />`https://american-american-football.sportdetect.com/standings`<br /><br />Get the standing from the `id`<br />`https://american-american-football.sportdetect.com/standings?id=eq.{id}`<br /><br />Get the standing from the `league_id`<br />`https://american-american-football.sportdetect.com/standings?league_id=eq.{league_id}`<br /><br />Get the standing from the `league_id` and `season_id`<br />`https://american-american-football.sportdetect.com/standings?league_id=eq.{league_id}&season_id=eq.{season_id}`<br /><br />Get the standing from the `league_id`, `season_id` and `type`<br />`https://american-american-football.sportdetect.com/standings?league_id=eq.{league_id}&season_id=eq.{season_id}&type=eq.{type}`"
    league_id: The id of the league. example:`eq.{league_id}`
        limit: Limiting and Pagination
        type: The type of the standing. example:`eq.{type}`
        season_id: The id of the season. example:`eq.{season_id}`
        is_id: The id of the standing. example:`eq.{id}`
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/standings"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    if season_id:
        querystring['season_id'] = season_id
    if is_id:
        querystring['id'] = is_id
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_events(event_id: str='eq.30199', limit: str='50', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all news of the events.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 15 minutes.<br />**Recommended Calls**: 1 call every 15 minutes.
		
		### Use Cases
		Get the news from the `event_id`<br />`https://american-american-football.sportdetect.com/news-events?event_id=eq.{event_id}`"
    event_id: The id of the event. example:`eq.{event_id}`
        limit: Limiting and Pagination
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/news-events"
    querystring = {}
    if event_id:
        querystring['event_id'] = event_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_teams(offset: str='0', team_id: str='eq.3478', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all news of the teams.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 15 minutes.<br />**Recommended Calls**: 1 call every 15 minutes.
		
		### Use Cases
		Get the news from the `team_id`<br />`https://american-american-football.sportdetect.com/news-teams?team_id=eq.{team_id}`"
    offset: Limiting and Pagination
        team_id: The id of the team. example:`eq.{team_id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/news-teams"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if team_id:
        querystring['team_id'] = team_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_players(player_id: str='eq.1569', limit: str='50', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all news of the players.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 15 minutes.<br />**Recommended Calls**: 1 call every 15 minutes.
		
		### Use Cases
		Get the news from the `player_id`<br />`https://american-american-football.sportdetect.com/news-players?player_id=eq.{player_id}`"
    player_id: The id of the player. example:`eq.{player_id}`
        limit: Limiting and Pagination
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/news-players"
    querystring = {}
    if player_id:
        querystring['player_id'] = player_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(choice_group: str=None, market_name: str='eq.Full time', offset: str='0', event_id: str='eq.3442', limit: str='50', is_live: str='eq.false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all odds from bet365.
		
		| market_name |
		| --- |
		| Full time |
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 5 minutes.<br />**Recommended Calls**: 1 call per 5 minutes.
		
		### Use Cases
		Get the odds by the `event_id`<br />`https://american-american-football.sportdetect.com/odds?event_id=eq.{event_id}`<br /><br />Get the odds by the `event_id` and `market_name`<br />`https://american-american-football.sportdetect.com/injuries?event_id=eq.{event_id}&market_name=eq.{market_name}`<br /><br />Get the odds by the `event_id`, `market_name` and `choice_group`<br />`https://american-american-football.sportdetect.com/injuries?event_id=eq.{event_id}&market_name=eq.{market_name}&choice_group=eq.{choice_group}`"
    choice_group: The group of the market. example:`eq.{choice_group}`
        market_name: The name of the market type. example:`eq.{market_name}`
        offset: Limiting and Pagination
        event_id: The id of the event. example:`eq.{event_id}`
        limit: Limiting and Pagination
        is_live: If the event is live. example:`eq.{is_live}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/odds"
    querystring = {}
    if choice_group:
        querystring['choice_group'] = choice_group
    if market_name:
        querystring['market_name'] = market_name
    if offset:
        querystring['offset'] = offset
    if event_id:
        querystring['event_id'] = event_id
    if limit:
        querystring['limit'] = limit
    if is_live:
        querystring['is_live'] = is_live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_transfers(offset: str='0', limit: str='50', transfer_to_team_id: str='eq.16979', player_id: str='eq.27423', transfer_from_team_id: str='eq.14850', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all player's transfers.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 1 hour.<br />**Recommended Calls**: 1 call per hour.
		
		### Use Cases
		Get the transfers from the `player_id`<br />`https://american-american-football.sportdetect.com/players-transfers?player_id=eq.{player_id}`<br /><br />Get the transfers from the `transfer_from_team_id` and  `transfer_to_team_id` ex.(team_id)<br />`https://american-american-football.sportdetect.com/players-transfers?or=(transfer_from_team_id.eq.{team_id},transfer_to_team_id.eq.{team_id})`<br /><br />"
    offset: Limiting and Pagination
        limit: Limiting and Pagination
        transfer_to_team_id: The id of the team that the player have been transferred to. example:`eq.{transfer_to_team_id}`
        player_id: The id of the player. example:`eq.{player_id}`
        transfer_from_team_id: The id of the team that the player have been transferred from. example:`eq.{transfer_from_team_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/players-transfers"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if transfer_to_team_id:
        querystring['transfer_to_team_id'] = transfer_to_team_id
    if player_id:
        querystring['player_id'] = player_id
    if transfer_from_team_id:
        querystring['transfer_from_team_id'] = transfer_from_team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players(team_id: str='eq.23042', offset: str='0', limit: str='50', is_id: str='eq.401982', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all players.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: Several times a day.
		
		You can get the image of the player by calling the following url: `https://images.sportdetect.com/{hash_image}.png`
		
		### Use Cases
		Get all players<br />`https://american-american-football.sportdetect.com/players`<br /><br />Search the players by the `name`<br />`https://american-american-football.sportdetect.com/players?name=like.*Cristiano*`<br /><br />Get the players from the `id`<br />`https://american-american-football.sportdetect.com/players?id=eq.{id}`<br /><br />Get the players from the `team_id`<br />`https://american-american-football.sportdetect.com/players?team_id=eq.{team_id}`"
    team_id: The id of the team. example:`eq.{team_id}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        is_id: The id of the player. example:`eq.{id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/players"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics(team_id: str='eq.3478', limit: str='50', type: str='eq.regularSeason', season_id: str='eq.748', player_id: str='eq.1569', offset: str='0', league_id: str='eq.828', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all player's statistics.
		
		For the type argument you can search by: **overall**, **home** and **away**, but **overall** is most common type for this endpoint.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 1 minute during a match.<br />**Recommended Calls**: 1 call per minute.
		
		### Use Cases
		Get the player statistics from the `player_id`<br />`https://american-american-football.sportdetect.com/players-statistics?player_id=eq.{player_id}`<br /><br />Get top 20 players with most goals from the `league_id` and `season_id`<br />`https://american-american-football.sportdetect.com/players-statistics?league_id=eq.{league_id}&season_id=eq.{season_id}&order=goals.desc&limit=20`<br /><br />Get top 20 players with most rating from the `league_id` and `season_id`<br />`https://american-american-football.sportdetect.com/players-statistics?league_id=eq.{league_id}&season_id=eq.{season_id}&order=rating.desc&limit=20`<br /><br />Get top 20 players with most red cards from the `league_id` and `season_id`<br />`https://american-american-football.sportdetect.com/players-statistics?league_id=eq.{league_id}&season_id=eq.{season_id}&order=red_cards.desc&limit=20`<br /><br />Get top 20 players with most yellow cards from the `league_id` and `season_id`<br />`https://american-american-football.sportdetect.com/players-statistics?league_id=eq.{league_id}&season_id=eq.{season_id}&order=yellow_cards.desc&limit=20`<br /><br />Get top 20 players with most assists from the `league_id` and `season_id`<br />`https://american-american-football.sportdetect.com/players-statistics?league_id=eq.{league_id}&season_id=eq.{season_id}&order=assists.desc&limit=20`<br /><br />Get the player statistics from the `player_id` and `team_id`<br />`https://american-american-football.sportdetect.com/players-statistics?player_id=eq.{player_id}&team_id=eq.{team_id}`<br /><br />Get the player statistics from the `player_id`, `team_id` and `league_id`<br />`https://american-american-football.sportdetect.com/players-statistics?player_id=eq.{player_id}&team_id=eq.{team_id}&league_id=eq.{league_id}`<br /><br />Get the player statistics from the `player_id`, `team_id`, `league_id` and `season_id`<br />`https://american-american-football.sportdetect.com/players-statistics?player_id=eq.{player_id}&team_id=eq.{team_id}&league_id=eq.{league_id}&season_id=eq.{season_id}`<br /><br />Get the player statistics from the `player_id`, `team_id`, `league_id`, `season_id` and `type`<br />`https://american-american-football.sportdetect.com/players-statistics?player_id=eq.{player_id}&team_id=eq.{team_id}&league_id=eq.{league_id}&season_id=eq.{season_id}&type=eq.{type}`"
    team_id: The id of the team. example:`eq.{team_id}`
        limit: Limiting and Pagination
        type: The type of the statistics. example:`eq.{type}`
        season_id: The id of the season. example:`eq.{season_id}`
        player_id: The id of the player. example:`eq.{player_id}`
        offset: Limiting and Pagination
        league_id: The id of the league. example:`eq.{league_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/players-statistics"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    if season_id:
        querystring['season_id'] = season_id
    if player_id:
        querystring['player_id'] = player_id
    if offset:
        querystring['offset'] = offset
    if league_id:
        querystring['league_id'] = league_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(limit: str='50', is_id: str='eq.1', alpha: str='eq.NX', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all countries.
		You can use the `alpha` field to get the country
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every hour.<br />**Recommended Calls**: 1 call per hour.
		
		You can get the image of the country by calling the following url: `https://images.sportdetect.com/{hash_image}.png`
		
		### Use Cases
		Get all countries<br />`https://american-american-football.sportdetect.com/countries`<br /><br />Get the country from the `id`<br />`https://american-american-football.sportdetect.com/countries?id=eq.{id}`<br /><br />Get all countries from the `alpha`<br />`https://american-american-football.sportdetect.com/countries?alpha=eq.{alpha}`"
    limit: Limiting and Pagination
        is_id: The id of the country. example:`eq.{id}`
        alpha: The alpha2 name of the country. example:`eq.{alpha}`
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/countries"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    if alpha:
        querystring['alpha'] = alpha
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cuptrees(season_id: str='eq.32271', offset: str='0', limit: str='50', league_id: str='eq.12165', is_id: str='eq.12482', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all cup knock outs.
		You can use `id` from the `seasons` or `leagues` endpoint to get the wanted cup knock out tree.
		
		Fields in rounds:
		
		Field | Explanation
		--- | ---
		`order` | The order of the round
		`description` | The description of the round
		`blocks` | The blocks of the round
		`finished` | If the event is finished
		`order` | The order of the block
		`home_team_score` | The score of the home team
		`away_team_score` | The score of the away team
		`has_next_round_link` | If the event has next round
		`event_in_progress` | If the event is in play
		`series_start_date_timestamp` | The start date of the event
		`automatic_progression` | If the event has automatic progression
		`participants` | The participants of the block (event)
		`team_id` | The id of the team
		`winner` | If the team is winner of the block
		`order` | The order of the participants
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated after every finished match.<br />**Recommended Calls**: 1 call after finished match.
		
		### Use Cases
		Get one cup knock out from the `id`<br />`https://american-american-football.sportdetect.com/cuptrees?id=eq.{id}`<br /><br />Get cups from the `season_id`<br />`https://american-american-football.sportdetect.com/cuptrees?season_id=eq.{season_id}`<br /><br />Get cups from the `league_id`<br />`https://american-american-football.sportdetect.com/cuptrees?league_id=eq.{league_id}`"
    season_id: The id of the season. example:`eq.{season_id}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        league_id: The id of the league. example:`eq.{league_id}`
        is_id: The id of the cup tree. example:`eq.{id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/cuptrees"
    querystring = {}
    if season_id:
        querystring['season_id'] = season_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if league_id:
        querystring['league_id'] = league_id
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def managers(team_id: str='eq.22962', country_id: str='eq.20', is_id: str='eq.12875', offset: str='0', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all managers.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per hour.
		
		You can get the image of the manager by calling the following url: `https://images.sportdetect.com/{hash_image}.png`
		
		### Use Cases
		Get the manager from the `id`<br />`https://american-american-football.sportdetect.com/managers?id=eq.{id}`<br /><br />Get the manager from the `country_id`<br />`https://american-american-football.sportdetect.com/managers?country_id=eq.{country_id}`<br /><br />Get the manager from the `team_id`<br />`https://american-american-football.sportdetect.com/managers?team_id=eq.{team_id}`"
    team_id: The id of the team. example:`eq.{team_id}`
        country_id: The id of the country. example:`eq.{country_id}`
        is_id: The id of the manager. example:`eq.{id}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/managers"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if country_id:
        querystring['country_id'] = country_id
    if is_id:
        querystring['id'] = is_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_leagues(offset: str='0', team_id: str='eq.3460', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all team's leagues.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: Several times a day.
		
		### Use Cases
		Get the  leagues from the `team_id`<br />`https://american-american-football.sportdetect.com/teams-leagues?team_id=eq.{team_id}`"
    offset: Limiting and Pagination
        team_id: The id of the team. example:`eq.{team_id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/teams-leagues"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if team_id:
        querystring['team_id'] = team_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(limit: str='50', category_id: str='eq.198', primary_league_id: str='eq.828', is_id: str='eq.86036', country_id: str='eq.20', venue_id: str='eq.13849', manager_id: str='eq.7327', offset: str='0', tournament_id: str='eq.1743', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: Limiting and Pagination
        category_id: The id of the category. example:`eq.{category_id}`
        primary_league_id: The id of the primary league
        is_id: The id of the team. example:`eq.{id}`
        country_id: The id of the country. example:`eq.{country_id}`
        venue_id: The id of the venue. example:`eq.{venue_id}`
        manager_id: The id of the manager. example:`eq.{manager_id}`
        offset: Limiting and Pagination
        tournament_id: The id of the tournament. example:`eq.{tournament_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/teams"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category_id:
        querystring['category_id'] = category_id
    if primary_league_id:
        querystring['primary_league_id'] = primary_league_id
    if is_id:
        querystring['id'] = is_id
    if country_id:
        querystring['country_id'] = country_id
    if venue_id:
        querystring['venue_id'] = venue_id
    if manager_id:
        querystring['manager_id'] = manager_id
    if offset:
        querystring['offset'] = offset
    if tournament_id:
        querystring['tournament_id'] = tournament_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_seasons(team_id: str='eq.3460', limit: str='50', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all team's seasons.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: Several times a day.
		
		### Use Cases
		Get the seasons from the `team_id`<br />`https://american-american-football.sportdetect.com/teams-seasons?team_id=eq.{team_id}`"
    team_id: The id of the team. example:`eq.{team_id}`
        limit: Limiting and Pagination
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/teams-seasons"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_info_last_season_attendance(league_id: str='eq.1', offset: str='0', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all last season attendance of the leagues and the cups.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call after every finished season.
		
		### Use Cases
		Get the last season attendance from the `league_id`<br />`https://american-american-football.sportdetect.com/leagues-info-last-season-attendance?league_id=eq.{league_id}`"
    league_id: The id of the league. example:`eq.{league_id}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/leagues-info-last-season-attendance"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(offset: str='0', category_id: str='eq.198', is_id: str='eq.12671', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all leagues and cups.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per hour.
		
		You can get the image of the league by calling the following url: `https://images.sportdetect.com/{hash_image}.png`
		
		### Use Cases
		Get all leagues<br />`https://american-american-football.sportdetect.com/leagues`<br /><br />Search the leagues by the `name`<br />`https://american-american-football.sportdetect.com/leagues?name=like.*Championship*`<br /><br />Get the leagues from the `id`<br />`https://american-american-football.sportdetect.com/leagues?id=eq.{id}`<br /><br />Get the leagues from the `category_id`<br />`https://american-american-football.sportdetect.com/leagues?category_id=eq.{category_id}`"
    offset: Limiting and Pagination
        category_id: The id of the category. example:`eq.{category_id}`
        is_id: The id of the league. example:`eq.{id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/leagues"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if category_id:
        querystring['category_id'] = category_id
    if is_id:
        querystring['id'] = is_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_info_last_season_top_scorers(offset: str='0', league_id: str='eq.1', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all last top scorers of the leagues and the cups.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call after every finished season.
		
		### Use Cases
		Get the top scorers from the `league_id`<br />`https://american-american-football.sportdetect.com/leagues-info-last-season-top-scorers?league_id=eq.{league_id}`"
    offset: Limiting and Pagination
        league_id: The id of the league. example:`eq.{league_id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/leagues-info-last-season-top-scorers"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if league_id:
        querystring['league_id'] = league_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_info_tv_partners(offset: str='0', limit: str='50', league_id: str='eq.1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all tv partners of the leagues and the cups.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per day.
		
		### Use Cases
		Get the tv partners from the `league_id`<br />`https://american-american-football.sportdetect.com/leagues-info-tv-partners?league_id=eq.{league_id}`"
    offset: Limiting and Pagination
        limit: Limiting and Pagination
        league_id: The id of the league. example:`eq.{league_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/leagues-info-tv-partners"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if league_id:
        querystring['league_id'] = league_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_info_promotions(league_id: str='eq.1', limit: str='50', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all promotions of the leagues and the cups.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per day.
		
		### Use Cases
		Get the promotions from the `league_id`<br />`https://american-american-football.sportdetect.com/leagues-info-promotions?league_id=eq.{league_id}`"
    league_id: The id of the league. example:`eq.{league_id}`
        limit: Limiting and Pagination
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/leagues-info-promotions"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_info_largest_stadium(offset: str='0', league_id: str='eq.1', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all largest stadiums of of the leagues and the cups.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per month.
		
		### Use Cases
		Get the largest stadium from the `league_id`<br />`https://american-american-football.sportdetect.com/leagues-info-largest-stadium?league_id=eq.{league_id}`"
    offset: Limiting and Pagination
        league_id: The id of the league. example:`eq.{league_id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/leagues-info-largest-stadium"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if league_id:
        querystring['league_id'] = league_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_info_official_organisation(league_id: str='eq.1', limit: str='50', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return official organisations of the leagues and the cups.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per day.
		
		### Use Cases
		Get the official organisation from the `league_id`<br />`https://american-american-football.sportdetect.com/leagues-info-official-organisation?league_id=eq.{league_id}`"
    league_id: The id of the league. example:`eq.{league_id}`
        limit: Limiting and Pagination
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/leagues-info-official-organisation"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_tv_channels(offset: str='0', alpha2: str='eq.AD', event_id: str='eq.1', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all tv channels for every event from all countries.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per hour.
		
		### Use Cases
		Get all tv channels for the event by the `event_id`<br />`https://american-american-football.sportdetect.com/events-tv-channels?event_id=eq.{event_id}`<br /><br />Get all tv channels for the event by the `event_id` and `alpha2`<br />`https://american-american-football.sportdetect.com/events-tv-channels?event_id=eq.{event_id}&alpha2=eq.{alpha2}`"
    offset: Limiting and Pagination
        alpha2: The alpha2 of the country. example:`eq.{alpha2}`
        event_id: The id of the event. example:`eq.{event_id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-tv-channels"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if alpha2:
        querystring['alpha2'] = alpha2
    if event_id:
        querystring['event_id'] = event_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(alpha: str='eq.US', offset: str='0', limit: str='50', is_id: str='eq.198', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all categories.
		You can use the `alpha` field to get the specific category as a country.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every hour.<br />**Recommended Calls**: 1 call per hour.
		
		You can get the image of the category by calling the following url: `https://images.sportdetect.com/{hash_image}.png`
		
		### Use Cases
		Get all categories<br />`https://american-american-football.sportdetect.com/categories`<br /><br />Get one category from the `id`<br />`https://american-american-football.sportdetect.com/categories?id=eq.{id}`<br /><br />Get all categories from the `alpha`<br />`https://american-american-football.sportdetect.com/categories?alpha=eq.{alpha}`"
    alpha: The alpha2 name of the category (_country_). example:`eq.{alpha}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        is_id: The id of the category. example:`eq.{id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/categories"
    querystring = {}
    if alpha:
        querystring['alpha'] = alpha
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def money(is_id: str='eq.85', offset: str='0', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all money values.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		### Use Cases
		Get the money from the `id`<br />`https://american-american-football.sportdetect.com/money?id=eq.{id}`"
    is_id: The id. example:`eq.{id}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/money"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tv_channels(offset: str='0', limit: str='50', is_id: str='eq.1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all tv channels for every country.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a week.<br />**Recommended Calls**: 1 call per hour.
		
		### Use Cases
		Get all tv channels<br />`https://american-american-football.sportdetect.com/tv-channels`<br /><br />Search the  tv channels by the `name`<br />`https://american-american-football.sportdetect.com/tv-channels?name=like.*Sportklub*`<br /><br />Get the tv channels by the `id`<br />`https://american-american-football.sportdetect.com/tv-channels?id=eq.{id}`"
    offset: Limiting and Pagination
        limit: Limiting and Pagination
        is_id: The id of the tv channel. example:`eq.{id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/tv-channels"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments(category_id: str='eq.198', is_id: str='eq.7325', league_id: str='eq.3096', offset: str='0', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all tournaments.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a week.<br />**Recommended Calls**: Several times a week.
		
		### Use Cases
		Get all tournaments<br />`https://american-american-football.sportdetect.com/tournaments`<br /><br />Get the tournaments from the `id`<br />`https://american-american-football.sportdetect.com/tournaments?id=eq.{id}`<br /><br />Get the tournaments from the `league_id`<br />`https://american-american-football.sportdetect.com/tournaments?league_id=eq.{league_id}`<br /><br />Get the tournaments from the `category_id`<br />`https://american-american-football.sportdetect.com/tournaments?category_id=eq.{category_id}`"
    category_id: The id of the category. example:`eq.{category_id}`
        is_id: The id of the tournament. example:`eq.{id}`
        league_id: The id of the league. example:`eq.{league_id}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/tournaments"
    querystring = {}
    if category_id:
        querystring['category_id'] = category_id
    if is_id:
        querystring['id'] = is_id
    if league_id:
        querystring['league_id'] = league_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_leagues(league_id: str='eq.4290', limit: str='50', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all medias for the leagues.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per hour.
		
		### Use Cases
		Get the media from the `league_id`<br />`https://american-american-football.sportdetect.com/media-leagues?league_id=eq.{league_id}`"
    league_id: The id of the league. example:`eq.{league_id}`
        limit: Limiting and Pagination
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/media-leagues"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geolocations(offset: str='0', is_id: str='eq.1', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return the geolocation of the venues.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated according to the venues.<br />**Recommended Calls**: 1 call per every venue.
		
		### Use Cases
		Get the geolocation from the `id`<br />`https://american-american-football.sportdetect.com/geolocations?id=eq.{id}`"
    offset: Limiting and Pagination
        is_id: The id of the location. example:`eq.{id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/geolocations"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if is_id:
        querystring['id'] = is_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues(offset: str='0', limit: str='50', is_id: str='eq.1', country_id: str='eq.3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all venues.
		With geolocation attribute from `geolocations` endpoint we can see latitude and longitute of the venue.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a week.<br />**Recommended Calls**: Several times a week.
		
		You can get the image of the venue by calling the following url: `https://images.sportdetect.com/{hash_image}.png`
		
		### Use Cases
		Get all venues<br />`https://american-american-football.sportdetect.com/venues`<br /><br />Get the venues from the `id`<br />`https://american-american-football.sportdetect.com/venues?id=eq.{id}`<br /><br />Get the venues from the `country_id`<br />`https://american-american-football.sportdetect.com/venues?country_id=eq.{country_id}`"
    offset: Limiting and Pagination
        limit: Limiting and Pagination
        is_id: The id of the venue. example:`eq.{id}`
        country_id: The id of the country. example:`eq.{country_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/venues"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_players(offset: str='0', limit: str='50', player_id: str='eq.290189', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all medias for the players.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: 1 call per hour.
		
		### Use Cases
		Get the media from the `player_id`<br />`https://american-american-football.sportdetect.com/media-players?player_id=eq.{player_id}`"
    offset: Limiting and Pagination
        limit: Limiting and Pagination
        player_id: The id of the player. example:`eq.{player_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/media-players"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if player_id:
        querystring['player_id'] = player_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_info(offset: str='0', season_id: str='eq.757', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all season's info.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 1 hour.<br />**Recommended Calls**: 1 call per hour.
		
		### Use Cases
		Get the info from the `season_id`<br />`https://american-american-football.sportdetect.com/seasons-info?season_id=eq.{season_id}`"
    offset: Limiting and Pagination
        season_id: The id of the season. example:`eq.{season_id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/seasons-info"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if season_id:
        querystring['season_id'] = season_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_rounds(limit: str='50', offset: str='0', season_id: str='eq.753', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all season's rounds.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 1 hour.<br />**Recommended Calls**: 1 call per hour.
		
		### Use Cases
		Get the rounds from the `season_id`<br />`https://american-american-football.sportdetect.com/seasons-rounds?season_id=eq.{season_id}`"
    limit: Limiting and Pagination
        offset: Limiting and Pagination
        season_id: The id of the season. example:`eq.{season_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/seasons-rounds"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(offset: str='0', limit: str='50', is_id: str='eq.757', league_id: str='eq.828', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all seasons.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 1 month.<br />**Recommended Calls**: 1 call per month.
		
		### Use Cases
		Get all seasons<br />`https://american-american-football.sportdetect.com/seasons`<br /><br />Get the seasons from the `id`<br />`https://american-american-football.sportdetect.com/seasons?id=eq.{id}`<br /><br />Get the seasons from the `league_id`<br />`https://american-american-football.sportdetect.com/seasons?league_id=eq.{league_id}`"
    offset: Limiting and Pagination
        limit: Limiting and Pagination
        is_id: The id of the season. example:`eq.{id}`
        league_id: The id of the league. example:`eq.{league_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/seasons"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    if league_id:
        querystring['league_id'] = league_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_managers(limit: str='50', offset: str='0', is_id: str='eq.1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all managers from the events.
		The events-manager's `id` is placed in `events` endpoint if it has the `managers_id` field.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: Several times a day.
		
		### Use Cases
		Get the manager from the `id`<br />`https://american-american-football.sportdetect.com/events-managers?id=eq.{id}`"
    limit: Limiting and Pagination
        offset: Limiting and Pagination
        is_id: The id of the event's managers. example:`eq.{id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-managers"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_lineups(offset: str='0', limit: str='50', is_id: str='eq.19', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all event's lineups.
		With this endpoint you can see the lineups from the event. This endpoint also has `confirmed` filed to check if the lineup is confirmed.
		
		Fields in lineup:
		
		Field | Explanation
		--- | ---
		`formation` | The formation of the lineup
		`player_color_primary` | The primary color of the players
		`player_color_number` | The number color of the players
		`player_color_outline` | The outline color of the players
		`goalkeeper_color_primary` | The primary color of the goalkeeper
		`goalkeeper_color_number` | The number color of the goalkeeper
		`goalkeeper_color_outline` | The outline color of the goalkeeper
		`players` | The players of the lineup
		
		Fields in players:
		
		Field | Explanation
		--- | ---
		`player_id` | The id of the player
		`shirt_number` | The shirt number of the player
		`jersey_number` | The jersey number of the player
		`position` | The position of the player
		`substitute` | If the player is substituted
		
		Resons for missing the match:
		
		Id | Reason
		--- | ---
		0 | Other
		1 | Injured
		2 | Ill
		3 | Suspended
		11 | YellowCard
		12 | YellowRedCard
		13 | RedCard
		21 | OnLoan
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.
		**Recommended Calls**: Several times a day.
		
		### Use Cases
		Get the lineups from the `id`<br />`https://american-american-football.sportdetect.com/events-lineups?id=eq.{id}`"
    offset: Limiting and Pagination
        limit: Limiting and Pagination
        is_id: The id of the lineup. example:`eq.{id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-lineups"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_statistics(limit: str='50', offset: str='0', event_id: str='eq.1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all statistics from the events.
		
		Types of the statistics:
		**Punts**, **First downs**, **Gross yards passing**, **First downs by rushing**, **Average yards per punt**, **Field goals**, **Touchdowns**, **Red zone efficiency**, **Fumbles lost**, **First downs by passing**, **Fumbles**, **Penalties**, **Passing - Interceptions**, **Rushing yards**, **Total yards**, **Third down efficiency**, **Net yards passing**, **Yards penalized**, **First downs by penalty**, **Time of Possession**, **Turnovers**.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times during a match.<br />**Recommended Calls**: Several times during a match.
		
		### Use Cases
		Get the statistics from the `event_id`<br />`https://american-american-football.sportdetect.com/events-statistics?event_id=eq.{event_id}`"
    limit: Limiting and Pagination
        offset: Limiting and Pagination
        event_id: The id of the event. example:`eq.{event_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-statistics"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if event_id:
        querystring['event_id'] = event_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_highlights(limit: str='50', offset: str='0', event_id: str='eq.2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all event's highlights.
		It has all social media posts about the event.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times during a match.<br />**Recommended Calls**: Several times during a match.
		
		### Use Cases
		Get the highlights from the `event_id`<br />`https://american-american-football.sportdetect.com/events-highlights?event_id=eq.{event_id}`"
    limit: Limiting and Pagination
        offset: Limiting and Pagination
        event_id: The id of the event
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-highlights"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if event_id:
        querystring['event_id'] = event_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_weather(limit: str='50', is_id: str='eq.1', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return the weather from the events.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times a day.<br />**Recommended Calls**: Several times a day.
		
		### Use Cases
		Get the weather from the `id`<br />`https://american-american-football.sportdetect.com/events-weather?id=eq.{id}`"
    limit: Limiting and Pagination
        is_id: The id of the weather. example:`eq.{id}`
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-weather"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_live(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all live events from the sport.
		
		It has the same results as the Events endpoint."
    
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_incidents(event_id: str='eq.1', offset: str='0', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all event's incidents.
		
		For **incidents** we have many types: `goal`, `period`.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times during a match.<br />**Recommended Calls**: Several times during a match.
		
		### Use Cases
		Get the incidents from the `event_id`<br />`https://american-american-football.sportdetect.com/events-incidents?event_id=eq.{event_id}`"
    event_id: The id of the event. example:`eq.{event_id}`
        offset: Limiting and Pagination
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-incidents"
    querystring = {}
    if event_id:
        querystring['event_id'] = event_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_coverage(event_id: str='eq.145', limit: str='50', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all event's coverages.
		With this endpoint you can see what data does your event has.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated after every hour.<br />**Recommended Calls**: 1 call after every match.
		
		### Use Cases
		Get the event coverage from the `event_id`<br />`https://american-american-football.sportdetect.com/events-coverage?event_id=eq.{event_id}`"
    event_id: The id of the event. example:`eq.{event_id}`
        limit: Limiting and Pagination
        offset: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-coverage"
    querystring = {}
    if event_id:
        querystring['event_id'] = event_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_players_statistics(offset: str='0', player_id: str='eq.159773', team_id: str='eq.3591', limit: str='50', event_id: str='eq.88', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all player's statistics from the events.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times during a match.<br />**Recommended Calls**: Several times during a match.
		
		### Use Cases
		Get the player statistics from the `event_id`<br />`https://american-american-football.sportdetect.com/events-players-statistics?event_id=eq.{event_id}`<br /><br />Get the best player in the event from the `event_id`<br />`https://american-american-football.sportdetect.com/events-players-statistics?event_id=eq.{event_id}&order=rating.desc&limit=1`<br /><br />Get the heatmaps from the `event_id` and `team_id`<br />`https://american-american-football.sportdetect.com/events-players-statistics?event_id=eq.{event_id}&team_id=eq.{team_id}`<br /><br />Get the heatmaps from the `event_id`, `team_id` and `player_id`<br />`https://american-american-football.sportdetect.com/events-players-statistics?event_id=eq.{event_id}&team_id=eq.{team_id}&player_id=eq.{player_id}`"
    offset: Limiting and Pagination
        player_id: The id of the player. example:`eq.{player_id}`
        team_id: The id of the team. example:`eq.{team_id}`
        limit: Limiting and Pagination
        event_id: The id of the event. example:`eq.{event_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-players-statistics"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if player_id:
        querystring['player_id'] = player_id
    if team_id:
        querystring['team_id'] = team_id
    if limit:
        querystring['limit'] = limit
    if event_id:
        querystring['event_id'] = event_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(venue_id: str='eq.5957', home_team_id: str='eq.86068', referee_id: str=None, offset: str='0', is_id: str='eq.30', tournament_id: str='eq.60027', limit: str='50', away_team_id: str='eq.86091', season_id: str='eq.31285', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all events.
		You can use `id` from the `seasons`, `tournaments`, `rounds`, `venues`, `referees` and `teams` endpoint to get the events.
		Events status codes:
		
		`type` | `description`
		--- | ---
		canceled | Canceled
		finished | Ended
		finished | AET
		finished | Awarded
		notstarted | Not started
		postponed | Postponed
		
		For **scores** we have fields for covering that: `current`, `display`, `period1`, `period2`, `period3`, `period4`, `normaltime`, `overtime`, `aggregated`.
		
		For **time extras or injuries** we have fields for covering that: **current_period_start_timestamp**.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		
		To switch pages you should use `/{endpoint}?limit={limit}&offset={offset}` ex.(`/{endpoint}?limit=15&offset=30}`) as parameters in endpoint.
		
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated every 10 seconds.<br />**Recommended Calls**: 1 call every 10 seconds.
		
		### Use Cases
		Get all events<br />`https://american-american-football.sportdetect.com/events`<br /><br />Get event from the `id`<br />`https://american-american-football.sportdetect.com/events?id=eq.{id}`<br /><br />Get events from the `season_id`<br />`https://american-american-football.sportdetect.com/events?season_id=eq.{season_id}`<br /><br />Get events from the `tournament_id`<br />`https://american-american-football.sportdetect.com/events?tournament_id=eq.{tournament_id}`<br /><br />Get events from the `round_id`<br />`https://american-american-football.sportdetect.com/events?round_id=eq.{round_id}`<br /><br />Get events from the `venue_id`<br />`https://american-american-football.sportdetect.com/events?venue_id=eq.{venue_id}`<br /><br />Get events from the `referee_id`<br />`https://american-american-football.sportdetect.com/events?referee_id=eq.{referee_id}`<br /><br />**H2H** - Get head to head events with teams ex.(home_team_id, away_team_id)<br />`https://american-american-football.sportdetect.com/events?or=(home_team_id.eq.{home_team_id}, away_team_id.eq.{home_team_id}, home_team_id.eq.{away_team_id}, away_team_id.eq.{away_team_id})`<br /><br />Get home team events from the `home_team_id`<br />`https://american-american-football.sportdetect.com/events?home_team_id=eq.{home_team_id}`<br /><br />Get away team events from the `away_team_id`<br />`https://american-american-football.sportdetect.com/events?away_team_id=eq.{away_team_id}`"
    venue_id: The id of the venue. example:`eq.{venue_id}`
        home_team_id: The id of the home team. example:`eq.{home_team_id}`
        referee_id: The id of the referee. example:`eq.{referee_id}`
        offset: Limiting and Pagination
        is_id: The id of the event. example:`eq.{id}`
        tournament_id: The id of the tournament. example:`eq.{tournament_id}`
        limit: Limiting and Pagination
        away_team_id: The id of the away team. example:`eq.{away_team_id}`
        season_id: The id of the season. example:`eq.{season_id}`
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events"
    querystring = {}
    if venue_id:
        querystring['venue_id'] = venue_id
    if home_team_id:
        querystring['home_team_id'] = home_team_id
    if referee_id:
        querystring['referee_id'] = referee_id
    if offset:
        querystring['offset'] = offset
    if is_id:
        querystring['id'] = is_id
    if tournament_id:
        querystring['tournament_id'] = tournament_id
    if limit:
        querystring['limit'] = limit
    if away_team_id:
        querystring['away_team_id'] = away_team_id
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_graphs(offset: str='0', is_id: str='eq.1', limit: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you can return all event's graphs.
		If the `value` field is a bigger number (> 0) that means that the home team had a big pressure that minute. And if the `value` field is a lower number (< 0) than it means that away team had a bigger pressure that minute. The events-graph's `id` is placed in `events` endpoint if it has the `graph_id` field.
		
		This endpoint uses a pagination system and the page argument allows you to switch between the various pages.
		> **Pagination**: 50 results per page. Enter offset and limit number to get the next results.
		
		**Update Period**: The endpoint is updated several times during a match.<br />**Recommended Calls**: Several times during a match.
		
		### Use Cases
		Get the graph from the `id`<br />`https://american-american-football.sportdetect.com/events-graphs?id=eq.{id}`"
    offset: Limiting and Pagination
        is_id: The id of the graph. example:`eq.{id}`
        limit: Limiting and Pagination
        
    """
    url = f"https://american-football-detect.p.rapidapi.com/events-graphs"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if is_id:
        querystring['id'] = is_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "american-football-detect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

