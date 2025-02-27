import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def odds_from_league_id_filtered_by_bookmaker(league_id: int, bookmaker_id: int, page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one league 
		you can filter by bookmaker or label like this : 
		- /v2/odds/league/{league_id}/bookmaker/{bookmaker_id}
		- /v2/odds/league/{league_id}/label/{label_id}
		
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    page: Use for the pagination (Default : 1)
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/league/{league_id}/bookmaker/{bookmaker_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_from_one_date(date: str, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one date
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/date/{date}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_head_to_head_between_two_teams(h2h: str, is_from: str=None, to: str=None, last: int=None, venue: int=None, season: int=None, date: str=None, next: int=None, status: str=None, timezone: str=None, league: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get heads to heads between two teams.
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here is an example of what can be achieved
		![demo-h2h](https://www.api-football.com/public/img/demo/demo-h2h.png)
		
		**Use Cases**
		Get all head to head between two {team}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?h2h=33-34`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?h2h=33-34`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?h2h=33-34&status=ns`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?h2h=33-34&from=2019-10-01&to=2019-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?league=39&season=2019&h2h=33-34&last=5`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?league=39&season=2019&h2h=33-34&next=10&from=2019-10-01&to=2019-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead?league=39&season=2019&h2h=33-34&last=5&timezone=Europe/London`"
    h2h: The ids of the teams `id-id`
        is_from: A valid date

        to: A valid date
        last: For the X last fixtures
`<= 2 characters`
        venue: The venue id of the fixture
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        date: A valid date
        next: For the X next fixtures
`<= 2 characters`
        status: One or more fixture status short
Like NS or NS-FT-CANC
        timezone: A valid timezone from the endpoint `Timezone`
        league: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/headtohead"
    querystring = {'h2h': h2h, }
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if last:
        querystring['last'] = last
    if venue:
        querystring['venue'] = venue
    if season:
        querystring['season'] = season
    if date:
        querystring['date'] = date
    if next:
        querystring['next'] = next
    if status:
        querystring['status'] = status
    if timezone:
        querystring['timezone'] = timezone
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_top_yellow_cards(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 20 players with the most yellow cards for a league or cup.
		
		**How it is calculated:**
		        * 1 : The player that received the higher number of yellow cards
		        * 2 : The player that received the higher number of red cards
		        * 3 : The player that assists in the higher number of matches
		        * 4 : The player that played the fewer minutes
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day."
    season: The season of the league
`4 characters` Like 2020, 2021 ...
        league: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players/topyellowcards"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_injuries_by_team_id_and_season(team: int=33, timezone: str=None, player: int=None, date: str=None, fixture: int=None, season: int=2020, league: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of players not participating in the fixtures for various reasons such as `suspended`, `injured` for example.
		Being a new endpoint, the data is only available from April 2021.
		
		**There are two types:**
		* `Missing Fixture` : The player will not play the fixture.
		* `Questionable` : The information is not yet 100% sure, the player may eventually play the fixture.
		
		> Examples available in Request samples "Use Cases".
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		**Update Frequency** : This endpoint is updated every 4 hours.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available injuries from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020`
		
		Get all available injuries from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?fixture=686314`
		
		Get all available injuries from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?team=85&season=2020`
		
		Get all available injuries from one {player} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?player=865&season=2020`
		
		Get all available injuries from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&player=865`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&timezone=Europe/London&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&league=61`"
    team: The id of the team
        timezone: A valid timezone from the endpoint `Timezone`
        player: The id of the player
        date: A valid date
        fixture: The id of the fixture
        season: The season of the league, **required** with `league`, `team` and `player` parameters
        league: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/injuries"
    querystring = {}
    if team:
        querystring['team'] = team
    if timezone:
        querystring['timezone'] = timezone
    if player:
        querystring['player'] = player
    if date:
        querystring['date'] = date
    if fixture:
        querystring['fixture'] = fixture
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_teams_seasons(team: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of seasons available for a team.
		
		**This endpoint requires at least one parameter.**
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all seasons available for a team from one team {id}
		`https://api-football-v1.p.rapidapi.com/v3/teams/seasons?team=33`"
    team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/seasons"
    querystring = {'team': team, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_in_play(bet: int=None, league: int=None, fixture: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns in-play odds for fixtures in progress.
		
		Fixtures are added between 15 and 5 minutes before the start of the fixture. Once the fixture is over they are removed from the endpoint between 5 and 20 minutes. **No history is stored**. So fixtures that are about to start, fixtures in progress and fixtures that have just ended are available in this endpoint.
		
		**Update Frequency** : This endpoint is updated every 5 seconds.`*`
		 `* This value can change in the range of 5 to 60 seconds`
		
		 **INFORMATIONS ABOUT STATUS**
		 ```
		        "status": {
		            "stopped": false, // True if the fixture is stopped by the referee for X reason
		            "blocked": false, // True if bets on this fixture are temporarily blocked
		            "finished": false // True if the fixture has not started or if it is finished
		        },
		 ```
		
		 **INFORMATIONS ABOUT VALUES**
		
		When several identical values exist for the same bet the `main` field is set to `True` for the bet being considered, the others will have the value `False`.
		
		The `main` field will be set to `True` only if several identical values exist for the same bet.
		
		When a value is unique for a bet the `main` value will always be `False` or `null`.
		
		**Example below** :
		        ```
		        "id": 36,
		        "name": "Over/Under Line",
		        "values": [
		            {
		                "value": "Over",
		                "odd": "1.975",
		                "handicap": "2",
		                "main": true, // Bet to consider
		                "suspended": false // True if this bet is temporarily suspended
		            },
		            {
		                "value": "Over",
		                "odd": "3.45",
		                "handicap": "2",
		                "main": false, // Bet to no consider
		                "suspended": false
		            },
		        ]"
    bet: The id of the bet
        league: The id of the league
`In this endpoint the \\\\\\\"season\\\\\\\" parameter is not needed`
        fixture: The id of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds/live"
    querystring = {}
    if bet:
        querystring['bet'] = bet
    if league:
        querystring['league'] = league
    if fixture:
        querystring['fixture'] = fixture
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_from_severals_fixtures_ids(is_id: int=None, to: str=None, team: int=None, round: str=None, is_from: str=None, league: int=None, timezone: str=None, last: int=None, season: int=None, live: str=None, ids: str='215662-215663-215664-215665-215666-215667', date: str=None, status: str=None, venue: int=None, next: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    is_id: The id of the fixture
        to: A valid date
        team: The id of the team
        round: The round of the fixture
        is_from: A valid date

        league: The id of the league
        timezone: A valid timezone from the endpoint `Timezone`
        last: For the X last fixtures
`<= 2 characters`
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        live:  Enum: `all` or `id-id` for filter by league id 
        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        date: A valid date
        status: One or more fixture status short
Like NS or NS-FT-CANC
        venue: The venue id of the fixture
        next: For the X next fixtures
`<= 2 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if to:
        querystring['to'] = to
    if team:
        querystring['team'] = team
    if round:
        querystring['round'] = round
    if is_from:
        querystring['from'] = is_from
    if league:
        querystring['league'] = league
    if timezone:
        querystring['timezone'] = timezone
    if last:
        querystring['last'] = last
    if season:
        querystring['season'] = season
    if live:
        querystring['live'] = live
    if ids:
        querystring['ids'] = ids
    if date:
        querystring['date'] = date
    if status:
        querystring['status'] = status
    if venue:
        querystring['venue'] = venue
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_coachs_by_team_id(search: str=None, is_id: int=None, team: int=33, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the information about the coachs and their careers.
		
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get coachs from one coach {id}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?id=1`
		
		Get coachs from one {team}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?team=33`
		
		Allows you to search for a coach in relation to a coach {name}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?search=Klopp`"
    search: The name of the coach
`>= 3 characters`
        is_id: The id of the coach
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/coachs"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coach_from_id(coach_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the information about the coachs and their careers
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    coach_id: coach_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/coachs/coach/{coach_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_round_for_a_league(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current round from one league or cup, All spaces are replaced by underscore “_”
		**Update Frequency** : Every day.
		**Recommended Calls** : 1 call per day."
    league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/rounds/{league_id}/current"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_from_player_id(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available transfers for players
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    player_id: player_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/transfers/player/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trophies_from_coach_id(coach_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available trophies for a coach
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    coach_id: coach_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/trophies/coach/{coach_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trophies_from_player_id(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available trophies for a players
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    player_id: player_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/trophies/player/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics_from_team_filtered_by_season(season: str, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all players statistics from one team filter by season
		Statistics are calculated according to the `team_id`, `league_id` and `season`
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    season: season (YYYY or YYYY-YYYY)
        team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/team/{team_id}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics_from_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all players statistics from one fixture
		**Update Frequency** : Every minute
		**Recommended Calls** : 1 call every minute for the fixtures in progress otherwise 1 call per day"
    fixture_id: fixture_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/fixture/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_statistics_filtered_by_date(league_id: int, team_id: int, date: str='2018-10-12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics for a one league/cup with a limit date
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the teams who have at least one fixture in progress otherwise 1 call per week"
    league_id: league_id
        team_id: team_id
        date: YYYY-MM-DD | produce : 2018-10-12 23:59:59
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/statistics/{league_id}/{team_id}/{date}"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_statistics(league_id: int, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics for a one league/cup
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the teams who have at least one fixture in progress otherwise 1 call per week"
    league_id: league_id
        team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/statistics/{league_id}/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league / cup
		**Update Frequency** : Every hour
		**Recommended Calls** : 1 call per hour for the leagues or teams who have at least one fixture in progress otherwise 1 call per day."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagueTable/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def labels_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available labels
		All labels `id` can be used in endpoint odds as filters
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/labels/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_from_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one fixture 
		you can filter by bookmaker or label like this : 
		- /v2/odds/fixture/{fixture_id}/bookmaker/{bookmaker_id}
		- /v2/odds/fixture/{fixture_id}/label/{label_id}
		
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/fixture/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available leagues"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_from_id(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one league from ID"
    league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/league/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_from_country(country_name: str, season: int=2018, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available leagues from country name or country code"
    country_name: country name or country code (ex : GB for England)
        season: Filter by season (2016, 2017, 2018, 2019)
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/country/{country_name}/{season}"
    querystring = {}
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available_seasons_for_a_league(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve all the seasons available in the API for a league/cup with the possibility to filter by {season}"
    league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/seasonsAvailable/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_seasons_for_all_leagues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve all the current seasons for all leagues and cups"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/current/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_by_type(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve league by type (league or cup)"
    type: the type of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/type/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_by_type_filtered_by_season(type: str, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve league by type (league or cup) and filtered by country"
    type: the type of the league
        season: the season of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/type/{type}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_availables(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All seasons are only **4-digit keys**, so for a league whose season is `2018-2019` like the English Premier League (EPL), the `2018-2019` season in the API will be `2018`.
		All `seasons` can be used in other endpoints as filters."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_from_id(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one team from ID
		The `team id` are unique in the API and teams keep it among all the leagues/cups in which they participate."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/teams/team/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_in_play(timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures in play
		In this request `events` are returned in the response
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/live"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_from_league_id(league_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one league or cup
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/league/{league_id}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_from_league_id_date(date: str, league_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one league or cup filter by date
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    date: date | YYYY-MM-DD
        league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/league/{league_id}/{date}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_from_league_id_round(league_id: int, round: str, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one league or cup filter by round, All spaces are replaced by underscore “_”
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    league_id: league_id
        round: round
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/league/{league_id}/{round}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def next_fixtures_from_league_id(number: int, league_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get X next fixtures from one league or cup
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    number: number
        league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/league/{league_id}/next/{number}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_fixtures_from_league_id(number: int, league_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get X last fixtures from one league or cup
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    number: number
        league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/league/{league_id}/last/{number}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def next_fixtures_from_team_id(number: int, team_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get x next fixtures from one team
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    number: number
        team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/team/{team_id}/next/{number}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_fixtures_from_team_id(team_id: int, number: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get x last fixtures from one team
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    team_id: team_id
        number: number
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/team/{team_id}/last/{number}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topscorers(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 20 best players for a league/cup
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/topscorers/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predictions(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get predictions about a fixture.
		The predictions are made using several algorithms including the poisson distribution, comparison of team statistics, last matches, players etc…"
    fixture_id: fixture_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/predictions/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available timezone to be used in the fixtures endpoint.
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available events from one fixture | Match events are updated at half-time and at the end of the match."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/events/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one fixture"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/odds/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_id_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one fixture from ID"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/fixtures/id/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics_league_id_team_id(team_id: int, league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics for a one league/cup."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/statistics/{league_id}/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_country_country_name_season(country_name: str, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available leagues from country name or country code"
    country_name: country name or country code (ex : GB for England)
        season: Filter by season (2016, 2017, 2018, 2019)
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/leagues/country/{country_name}/{season}"
    querystring = {}
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics_league_id_team_id_date(team_id: int, league_id: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics for a one league/cup with a limit date"
    date: YYYY-MM-DD | produce : 2018-10-12 23:59:59
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/statistics/{league_id}/{team_id}/{date}"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_season_team_id(team_id: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get coach and players from one team and season"
    season: 2016, 2017, 2018, 2019
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/players/{season}/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaguetable_league_id(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get standings from one league or cup."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/leagueTable/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics_fixture_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all statistics game from one fixture"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/statistics/fixture/{fixture_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_bets_in_play(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets for in-play odds.
		
		All bets `id` can be used in endpoint `odds/live` as filters, **but are not compatible with endpoint `odds` for pre-match odds**.
		
		**Update Frequency** : This endpoint is updated every 60 seconds."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds/live/bets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_next_x_fixtures_to_come(is_from: str=None, ids: str=None, league: int=None, to: str=None, round: str=None, live: str=None, venue: int=None, date: str=None, last: int=None, season: int=None, timezone: str=None, team: int=None, next: int=50, is_id: int=None, status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    is_from: A valid date

        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        league: The id of the league
        to: A valid date
        round: The round of the fixture
        live:  Enum: `all` or `id-id` for filter by league id 
        venue: The venue id of the fixture
        date: A valid date
        last: For the X last fixtures
`<= 2 characters`
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        timezone: A valid timezone from the endpoint `Timezone`
        team: The id of the team
        next: For the X next fixtures
`<= 2 characters`
        is_id: The id of the fixture
        status: One or more fixture status short
Like NS or NS-FT-CANC
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if ids:
        querystring['ids'] = ids
    if league:
        querystring['league'] = league
    if to:
        querystring['to'] = to
    if round:
        querystring['round'] = round
    if live:
        querystring['live'] = live
    if venue:
        querystring['venue'] = venue
    if date:
        querystring['date'] = date
    if last:
        querystring['last'] = last
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    if team:
        querystring['team'] = team
    if next:
        querystring['next'] = next
    if is_id:
        querystring['id'] = is_id
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_filtered_by_status(ids: str=None, venue: int=None, season: int=2020, is_from: str=None, round: str=None, next: int=None, to: str=None, status: str='FT', live: str=None, last: int=None, date: str=None, is_id: int=None, league: int=39, team: int=None, timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        venue: The venue id of the fixture
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        is_from: A valid date

        round: The round of the fixture
        next: For the X next fixtures
`<= 2 characters`
        to: A valid date
        status: One or more fixture status short
Like NS or NS-FT-CANC
        live:  Enum: `all` or `id-id` for filter by league id 
        last: For the X last fixtures
`<= 2 characters`
        date: A valid date
        is_id: The id of the fixture
        league: The id of the league
        team: The id of the team
        timezone: A valid timezone from the endpoint `Timezone`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if venue:
        querystring['venue'] = venue
    if season:
        querystring['season'] = season
    if is_from:
        querystring['from'] = is_from
    if round:
        querystring['round'] = round
    if next:
        querystring['next'] = next
    if to:
        querystring['to'] = to
    if status:
        querystring['status'] = status
    if live:
        querystring['live'] = live
    if last:
        querystring['last'] = last
    if date:
        querystring['date'] = date
    if is_id:
        querystring['id'] = is_id
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_filtered_by_round(ids: str=None, live: str=None, last: int=None, to: str=None, venue: int=None, season: int=2020, date: str=None, status: str=None, round: str='Regular Season - 10', is_from: str=None, next: int=None, is_id: int=None, team: int=None, timezone: str=None, league: int=39, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        live:  Enum: `all` or `id-id` for filter by league id 
        last: For the X last fixtures
`<= 2 characters`
        to: A valid date
        venue: The venue id of the fixture
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        date: A valid date
        status: One or more fixture status short
Like NS or NS-FT-CANC
        round: The round of the fixture
        is_from: A valid date

        next: For the X next fixtures
`<= 2 characters`
        is_id: The id of the fixture
        team: The id of the team
        timezone: A valid timezone from the endpoint `Timezone`
        league: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if live:
        querystring['live'] = live
    if last:
        querystring['last'] = last
    if to:
        querystring['to'] = to
    if venue:
        querystring['venue'] = venue
    if season:
        querystring['season'] = season
    if date:
        querystring['date'] = date
    if status:
        querystring['status'] = status
    if round:
        querystring['round'] = round
    if is_from:
        querystring['from'] = is_from
    if next:
        querystring['next'] = next
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    if timezone:
        querystring['timezone'] = timezone
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_between_two_dates(status: str=None, venue: int=None, ids: str=None, to: str='2021-04-07', league: int=39, team: int=None, timezone: str=None, next: int=None, date: str=None, last: int=None, season: int=2020, is_id: int=None, live: str=None, round: str=None, is_from: str='2021-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    status: One or more fixture status short
Like NS or NS-FT-CANC
        venue: The venue id of the fixture
        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        to: A valid date
        league: The id of the league
        team: The id of the team
        timezone: A valid timezone from the endpoint `Timezone`
        next: For the X next fixtures
`<= 2 characters`
        date: A valid date
        last: For the X last fixtures
`<= 2 characters`
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        is_id: The id of the fixture
        live:  Enum: `all` or `id-id` for filter by league id 
        round: The round of the fixture
        is_from: A valid date

        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if status:
        querystring['status'] = status
    if venue:
        querystring['venue'] = venue
    if ids:
        querystring['ids'] = ids
    if to:
        querystring['to'] = to
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if timezone:
        querystring['timezone'] = timezone
    if next:
        querystring['next'] = next
    if date:
        querystring['date'] = date
    if last:
        querystring['last'] = last
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if live:
        querystring['live'] = live
    if round:
        querystring['round'] = round
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures(round: str=None, ids: str=None, league: int=None, venue: int=None, season: int=None, is_from: str=None, live: str=None, team: int=None, is_id: int=None, date: str='2021-01-29', timezone: str=None, last: int=None, next: int=None, to: str=None, status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    round: The round of the fixture
        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        league: The id of the league
        venue: The venue id of the fixture
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        is_from: A valid date

        live:  Enum: `all` or `id-id` for filter by league id 
        team: The id of the team
        is_id: The id of the fixture
        date: A valid date
        timezone: A valid timezone from the endpoint `Timezone`
        last: For the X last fixtures
`<= 2 characters`
        next: For the X next fixtures
`<= 2 characters`
        to: A valid date
        status: One or more fixture status short
Like NS or NS-FT-CANC
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if round:
        querystring['round'] = round
    if ids:
        querystring['ids'] = ids
    if league:
        querystring['league'] = league
    if venue:
        querystring['venue'] = venue
    if season:
        querystring['season'] = season
    if is_from:
        querystring['from'] = is_from
    if live:
        querystring['live'] = live
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if date:
        querystring['date'] = date
    if timezone:
        querystring['timezone'] = timezone
    if last:
        querystring['last'] = last
    if next:
        querystring['next'] = next
    if to:
        querystring['to'] = to
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_by_league_id(team: int=None, ids: str=None, is_from: str=None, to: str=None, last: int=None, live: str=None, venue: int=None, is_id: int=None, timezone: str=None, date: str=None, next: int=None, season: int=2020, status: str=None, league: int=39, round: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    team: The id of the team
        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        is_from: A valid date

        to: A valid date
        last: For the X last fixtures
`<= 2 characters`
        live:  Enum: `all` or `id-id` for filter by league id 
        venue: The venue id of the fixture
        is_id: The id of the fixture
        timezone: A valid timezone from the endpoint `Timezone`
        date: A valid date
        next: For the X next fixtures
`<= 2 characters`
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        status: One or more fixture status short
Like NS or NS-FT-CANC
        league: The id of the league
        round: The round of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if team:
        querystring['team'] = team
    if ids:
        querystring['ids'] = ids
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if last:
        querystring['last'] = last
    if live:
        querystring['live'] = live
    if venue:
        querystring['venue'] = venue
    if is_id:
        querystring['id'] = is_id
    if timezone:
        querystring['timezone'] = timezone
    if date:
        querystring['date'] = date
    if next:
        querystring['next'] = next
    if season:
        querystring['season'] = season
    if status:
        querystring['status'] = status
    if league:
        querystring['league'] = league
    if round:
        querystring['round'] = round
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_in_progress_livescore(ids: str=None, is_from: str=None, next: int=None, status: str=None, venue: int=None, league: int=None, team: int=None, to: str=None, round: str=None, timezone: str=None, live: str='all', date: str=None, last: int=None, season: int=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        is_from: A valid date

        next: For the X next fixtures
`<= 2 characters`
        status: One or more fixture status short
Like NS or NS-FT-CANC
        venue: The venue id of the fixture
        league: The id of the league
        team: The id of the team
        to: A valid date
        round: The round of the fixture
        timezone: A valid timezone from the endpoint `Timezone`
        live:  Enum: `all` or `id-id` for filter by league id 
        date: A valid date
        last: For the X last fixtures
`<= 2 characters`
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        is_id: The id of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if is_from:
        querystring['from'] = is_from
    if next:
        querystring['next'] = next
    if status:
        querystring['status'] = status
    if venue:
        querystring['venue'] = venue
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if to:
        querystring['to'] = to
    if round:
        querystring['round'] = round
    if timezone:
        querystring['timezone'] = timezone
    if live:
        querystring['live'] = live
    if date:
        querystring['date'] = date
    if last:
        querystring['last'] = last
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_by_fixture_id(venue: int=None, last: int=None, team: int=None, next: int=None, ids: str=None, timezone: str=None, season: int=None, is_from: str=None, live: str=None, is_id: int=157201, league: int=None, to: str=None, status: str=None, date: str=None, round: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    venue: The venue id of the fixture
        last: For the X last fixtures
`<= 2 characters`
        team: The id of the team
        next: For the X next fixtures
`<= 2 characters`
        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        timezone: A valid timezone from the endpoint `Timezone`
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        is_from: A valid date

        live:  Enum: `all` or `id-id` for filter by league id 
        is_id: The id of the fixture
        league: The id of the league
        to: A valid date
        status: One or more fixture status short
Like NS or NS-FT-CANC
        date: A valid date
        round: The round of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if venue:
        querystring['venue'] = venue
    if last:
        querystring['last'] = last
    if team:
        querystring['team'] = team
    if next:
        querystring['next'] = next
    if ids:
        querystring['ids'] = ids
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    if is_from:
        querystring['from'] = is_from
    if live:
        querystring['live'] = live
    if is_id:
        querystring['id'] = is_id
    if league:
        querystring['league'] = league
    if to:
        querystring['to'] = to
    if status:
        querystring['status'] = status
    if date:
        querystring['date'] = date
    if round:
        querystring['round'] = round
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_teams_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of countries available for the `teams` endpoint.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all countries available for the teams endpoints
		`https://api-football-v1.p.rapidapi.com/v3/teams/countries`"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_teams_informations(venue: int=None, code: str=None, search: str=None, is_id: int=33, season: int=None, name: str=None, country: str=None, league: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available teams.
		
		The team `id` are **unique** in the API and teams keep it among all the leagues/cups in which they participate.
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get one team from one team {id}
		`https://api-football-v1.p.rapidapi.com/v3/teams?id=33`
		
		Get one team from one team {name}
		`https://api-football-v1.p.rapidapi.com/v3/teams?name=manchester united`
		
		Get all teams from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/teams?league=39&season=2019`
		
		Get teams from one team {country}
		`https://api-football-v1.p.rapidapi.com/v3/teams?country=england`
		
		Allows you to search for a team in relation to a team {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/teams?search=manches`
		`https://api-football-v1.p.rapidapi.com/v3/teams?search=England`"
    venue: The id of the venue
        code: The code of the team
        search: The name or the country name of the team
`>= 3 characters`
        is_id: The id of the team
        season: The season of the league
`4 characters` Like 2019, 2020, 2021 ...

        name: The name of the team
        country: The country name of the team
        league: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams"
    querystring = {}
    if venue:
        querystring['venue'] = venue
    if code:
        querystring['code'] = code
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if country:
        querystring['country'] = country
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_team(country: str=None, season: int=None, name: str=None, search: str='manches', league: int=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available teams.
		
		The team `id` are **unique** in the API and teams keep it among all the leagues/cups in which they participate.
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get one team from one team {id}
		`https://api-football-v1.p.rapidapi.com/v3/teams?id=33`
		
		Get one team from one team {name}
		`https://api-football-v1.p.rapidapi.com/v3/teams?name=manchester united`
		
		Get all teams from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/teams?league=39&season=2019`
		
		Get teams from one team {country}
		`https://api-football-v1.p.rapidapi.com/v3/teams?country=england`
		
		Allows you to search for a team in relation to a team {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/teams?search=manches`
		`https://api-football-v1.p.rapidapi.com/v3/teams?search=England`"
    country: The country name of the team
        season: The season of the league
`4 characters` Like 2019, 2020, 2021 ...

        name: The name of the team
        search: The name or the country name of the team
`>= 3 characters`
        league: The id of the league
        is_id: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams"
    querystring = {}
    if country:
        querystring['country'] = country
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_league(country: str=None, code: str=None, is_id: int=None, name: str=None, search: str='major', season: int=None, type: str=None, team: int=None, last: int=None, current: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    country: The country name of the league
        code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        is_id: The id of the league
        name: The name of the league
        search: The name or the country of the league
`>= 3 characters`
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        type: The type of the league 
 Enum: `league` or `cup` 
        team: The id of the team
        last: The X last leagues/cups added in the API
`<= 2 characters`
        current: The state of the league
Enum: `true` or `false` 
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if country:
        querystring['country'] = country
    if code:
        querystring['code'] = code
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if season:
        querystring['season'] = season
    if type:
        querystring['type'] = type
    if team:
        querystring['team'] = team
    if last:
        querystring['last'] = last
    if current:
        querystring['current'] = current
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_player(page: int=None, season: int=None, league: int=61, team: int=None, is_id: int=None, search: str='neymar', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players statistics.
		
		The players `id` are unique in the API and players keep it among all the teams they have been in
		
		The statistics are calculated according to the team `id`, league `id` and `season`.
		
		You can find the available `seasons` by using the endpoint players seasons.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages thanks to the `page` parameter.
		
		> **Pagination** : 20 results per page.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all players statistics from one player {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?id=19088&season=2018`
		
		Get all players statistics from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33&page=2`
		
		Get all players statistics from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&page=4`
		
		Get all players statistics from one {league}, {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33&page=5`
		
		Allows you to search for a player in relation to a player {name}
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?league=61&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani&season=2018`"
    page: Use for the pagination
Default: `1`
        season: The season of the league `4 characters`
Requires the fields `Id`, `League` or `Team`
        league: The id of the league
        team: The id of the team
        is_id: The id of the player
        search: The name of the player `>= 4 characters`
Requires the fields `League` or `Team`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players"
    querystring = {}
    if page:
        querystring['page'] = page
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_coach(team: int=None, search: str='tuche', is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the information about the coachs and their careers.
		
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get coachs from one coach {id}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?id=1`
		
		Get coachs from one {team}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?team=33`
		
		Allows you to search for a coach in relation to a coach {name}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?search=Klopp`"
    team: The id of the team
        search: The name of the coach
`>= 3 characters`
        is_id: The id of the coach
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/coachs"
    querystring = {}
    if team:
        querystring['team'] = team
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_bookmaker(search: str='bwi', is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers.
		
		All bookmakers `id` can be used in endpoint odds as filters.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available bookmakers
		`https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers`
		
		Get bookmaker from one {id}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers?id=1`
		
		Allows you to search for a bookmaker in relation to a bookmakers {name}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers?search=Betfair`"
    search: The name of the bookmaker
        is_id: The id of the bookmaker
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_country(name: str=None, search: str='engl', code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		
		The `name` and `code` fields can be used in other endpoints as filters.
		> Examples available in Request samples "Use Cases".
		
		All the parameters of this endpoint can be used together.
		**Update Frequency** : This endpoint is updated each time a new league from a country not covered by the API is added.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available countries across all {seasons} and competitions
		`https://api-football-v1.p.rapidapi.com/v3/countries`
		
		Get all available countries from one country {name}
		`https://api-football-v1.p.rapidapi.com/v3/countries?name=england`
		
		Get all available countries from one country {code}
		`https://api-football-v1.p.rapidapi.com/v3/countries?code=fr`
		
		Allows you to search for a countries in relation to a country {name}
		`https://api-football-v1.p.rapidapi.com/v3/countries?search=engl`"
    name: The name of the country

        search: The name of the country
`>= 3 characters`
        code: The Alpha2 code of the country
`2 characters` Like  FR, GB, IT… 
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/countries"
    querystring = {}
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_venue(name: str=None, is_id: int=None, city: str=None, search: str='old tr', country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available venues.
		
		The venue `id` are **unique** in the API.
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get one venue from venue {id}
		`https://api-football-v1.p.rapidapi.com/v3/venues?id=556`
		
		Get one venue from venue {name}
		`https://api-football-v1.p.rapidapi.com/v3/venues?name=Old Trafford`
		
		Get all venues from {city}
		`https://api-football-v1.p.rapidapi.com/v3/venues?city=manchester`
		
		Get venues from {country}
		`https://api-football-v1.p.rapidapi.com/v3/venues?country=england`
		
		Allows you to search for a venues in relation to a venue {name}, {city} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=trafford`
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=manches`
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=England`"
    name: The name of the venue
        is_id: The id of the venue
        city: The city of the venue
        search: The name, city or the country of the venue
`>= 3 characters`
        country: The country name of the venue
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/venues"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if city:
        querystring['city'] = city
    if search:
        querystring['search'] = search
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_search_bet(search: str='winner', is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets.
		
		All bets `id` can be used in endpoint odds as filters.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available bets
		`https://api-football-v1.p.rapidapi.com/v3/odds/bets`
		
		Get bet from one {id}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bets?id=1`
		
		Allows you to search for a bet in relation to a bets {name}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bets?search=winner`"
    search: The name of the bet
        is_id: The id of the bet
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds/bets"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available seasons.
		
		All seasons are only **4-digit keys**, so for a league whose season is `2018-2019` like the English Premier League (EPL), the `2018-2019` season in the API will be `2018`.
		
		All `seasons` can be used in other endpoints as filters.
		
		> This endpoint does not require any parameters.
		
		**Update Frequency** : This endpoint is updated each time a new league is added.
		**Recommended Calls** : 1 call per day."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_countries(code: str=None, search: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		
		The `name` and `code` fields can be used in other endpoints as filters.
		> Examples available in Request samples "Use Cases".
		
		All the parameters of this endpoint can be used together.
		**Update Frequency** : This endpoint is updated each time a new league from a country not covered by the API is added.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available countries across all {seasons} and competitions
		`https://api-football-v1.p.rapidapi.com/v3/countries`
		
		Get all available countries from one country {name}
		`https://api-football-v1.p.rapidapi.com/v3/countries?name=england`
		
		Get all available countries from one country {code}
		`https://api-football-v1.p.rapidapi.com/v3/countries?code=fr`
		
		Allows you to search for a countries in relation to a country {name}
		`https://api-football-v1.p.rapidapi.com/v3/countries?search=engl`"
    code: The Alpha2 code of the country
`2 characters` Like  FR, GB, IT… 
        search: The name of the country
`>= 3 characters`
        name: The name of the country

        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/countries"
    querystring = {}
    if code:
        querystring['code'] = code
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_venues_by_venue_id(city: str=None, is_id: int=556, country: str=None, name: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available venues.
		
		The venue `id` are **unique** in the API.
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get one venue from venue {id}
		`https://api-football-v1.p.rapidapi.com/v3/venues?id=556`
		
		Get one venue from venue {name}
		`https://api-football-v1.p.rapidapi.com/v3/venues?name=Old Trafford`
		
		Get all venues from {city}
		`https://api-football-v1.p.rapidapi.com/v3/venues?city=manchester`
		
		Get venues from {country}
		`https://api-football-v1.p.rapidapi.com/v3/venues?country=england`
		
		Allows you to search for a venues in relation to a venue {name}, {city} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=trafford`
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=manches`
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=England`"
    city: The city of the venue
        is_id: The id of the venue
        country: The country name of the venue
        name: The name of the venue
        search: The name, city or the country of the venue
`>= 3 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/venues"
    querystring = {}
    if city:
        querystring['city'] = city
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_mapping(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available odds.
		
		All fixtures, leagues `id` and `date` can be used in endpoint odds as filters.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages thanks to the `page` parameter.
		
		> **Pagination** : 100 results per page.
		
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day."
    page: Use for the pagination
 Default: `1`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds/mapping"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_statistics(fixture: int, team: int=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics for one fixture.
		
		**Available statistics**
		        * Shots on Goal
		        * Shots off Goal
		        * Shots insidebox
		        * Shots outsidebox
		        * Total Shots
		        * Blocked Shots
		        * Fouls
		        * Corner Kicks
		        * Offsides
		        * Ball Possession
		        * Yellow Cards
		        * Red Cards
		        * Goalkeeper Saves
		        * Total passes
		        * Passes accurate
		        * Passes %
		
		**Update Frequency** : This endpoint is updated every minute.
		**Recommended Calls** : 1 call every minute for the teams or fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here is an example of what can be achieved
		![demo-statistics](https://www.api-football.com/public/img/demo/demo-statistics.png)
		
		**Use Cases**
		Get all available statistics from one {fixture}
		`https://v3.football.api-sports.io/fixtures/statistics?fixture=215662`
		
		Get all available statistics from one {fixture} & {type}
		`https://v3.football.api-sports.io/fixtures/statistics?fixture=215662&type=Total Shots`
		
		Get all available statistics from one {fixture} & {team}
		`v3.football.api-sports.io/fixtures/statistics?fixture=215662&team=463`"
    fixture: The id of the fixture
        team: The id of the team
        type: The type of statistics
Like `Fouls`, `Offsides`...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_lineups(fixture: int, player: int=None, team: int=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the lineups for a fixture.
		
		**Available datas**
		        * Formation
		        * Coach
		        * Start XI
		        * Substitutes
		
		> Lineups are available between 20 and 40 minutes before the fixture.
		
		**Players' positions on the grid**
		
		**X** = row and **Y** = column (X:Y)
		
		Line 1 **X** being the one of the goal and then for each line this number is incremented. The column **Y** will go from left to right, and incremented for each player of the line.
		
		`As a new feature, some irregularities may occur, do not hesitate to report them on our public Roadmap`
		
		**Update Frequency** : This endpoint is updated every 15 minutes.
		**Recommended Calls** : 1 call every 15 minutes for the fixtures in progress otherwise 1 call per day.
		
		> Here are several examples of what can be done
		![demo-lineups](https://www.api-football.com/public/img/demo/demo-lineups-1.jpg)
		![demo-lineups](https://www.api-football.com/public/img/demo/demo-lineups.png)
		
		**Use Cases**
		Get all available lineups from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662`
		
		Get all available lineups from one {fixture} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&team=463`
		
		Get all available lineups from one {fixture} & {player}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&player=35845`
		
		Get all available lineups from one {fixture} & {type}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&type=startXI`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&player=35845&type=startXI`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&team=463&type=startXI&player=35845`"
    fixture: The id of the fixture
        player: The id of the player
        team: The id of the team
        type: The Lineup type
Like `Formation`, `Substitutes`...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups"
    querystring = {'fixture': fixture, }
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_players_statistics_by_fixture_id(fixture: int, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players statistics from one fixture.
		
		**Update Frequency** : This endpoint is updated every minute.
		**Recommended Calls** : 1 call every minute for the fixtures in progress otherwise 1 call per day.
		
		**Use Cases**
		Get all available players statistics from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/players?fixture=169080`
		
		Get all available players statistics from one {fixture} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/players?fixture=169080&team=2284`"
    fixture: The id of the fixture
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/players"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_players_statistics_by_player_id(is_id: int=276, search: str=None, season: int=2020, team: int=None, league: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players statistics.
		
		The players `id` are unique in the API and players keep it among all the teams they have been in
		
		The statistics are calculated according to the team `id`, league `id` and `season`.
		
		You can find the available `seasons` by using the endpoint players seasons.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages thanks to the `page` parameter.
		
		> **Pagination** : 20 results per page.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all players statistics from one player {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?id=19088&season=2018`
		
		Get all players statistics from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33&page=2`
		
		Get all players statistics from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&page=4`
		
		Get all players statistics from one {league}, {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33&page=5`
		
		Allows you to search for a player in relation to a player {name}
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?league=61&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani&season=2018`"
    is_id: The id of the player
        search: The name of the player `>= 4 characters`
Requires the fields `League` or `Team`
        season: The season of the league `4 characters`
Requires the fields `Id`, `League` or `Team`
        team: The id of the team
        league: The id of the league
        page: Use for the pagination
Default: `1`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    if season:
        querystring['season'] = season
    if team:
        querystring['team'] = team
    if league:
        querystring['league'] = league
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_sidelined_by_player_id(coach: int=None, player: int=276, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available sidelined for a player or a coach.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all from one {player}
		`https://api-football-v1.p.rapidapi.com/v3/sidelined?player=276`
		
		Get all from one {coach}
		`https://api-football-v1.p.rapidapi.com/v3/sidelined?coach=2`"
    coach: The id of the coach
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/sidelined"
    querystring = {}
    if coach:
        querystring['coach'] = coach
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_players_squads(team: int=33, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the current squad of a team when the `team` parameter is used. When the `player` parameter is used the endpoint returns the set of teams associated with the player.
		
		> The response format is the same regardless of the parameter sent.
		
		**This endpoint requires at least one parameter.**
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per week.
		
		**Use Cases**
		Get all players from one {team}
		`https://api-football-v1.p.rapidapi.com/v3/players/squads?team=33`
		
		Get all teams from one {player}
		`https://api-football-v1.p.rapidapi.com/v3/players/squads?player=276`"
    team: The id of the team
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players/squads"
    querystring = {}
    if team:
        querystring['team'] = team
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_injuries_by_player_id_and_season(season: int=2020, fixture: int=None, team: int=None, date: str=None, league: int=None, timezone: str=None, player: int=276, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of players not participating in the fixtures for various reasons such as `suspended`, `injured` for example.
		Being a new endpoint, the data is only available from April 2021.
		
		**There are two types:**
		* `Missing Fixture` : The player will not play the fixture.
		* `Questionable` : The information is not yet 100% sure, the player may eventually play the fixture.
		
		> Examples available in Request samples "Use Cases".
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		**Update Frequency** : This endpoint is updated every 4 hours.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available injuries from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020`
		
		Get all available injuries from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?fixture=686314`
		
		Get all available injuries from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?team=85&season=2020`
		
		Get all available injuries from one {player} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?player=865&season=2020`
		
		Get all available injuries from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&player=865`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&timezone=Europe/London&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&league=61`"
    season: The season of the league, **required** with `league`, `team` and `player` parameters
        fixture: The id of the fixture
        team: The id of the team
        date: A valid date
        league: The id of the league
        timezone: A valid timezone from the endpoint `Timezone`
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/injuries"
    querystring = {}
    if season:
        querystring['season'] = season
    if fixture:
        querystring['fixture'] = fixture
    if team:
        querystring['team'] = team
    if date:
        querystring['date'] = date
    if league:
        querystring['league'] = league
    if timezone:
        querystring['timezone'] = timezone
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_injuries_by_league_id_and_season(date: str=None, team: int=None, player: int=None, timezone: str=None, fixture: int=None, league: int=61, season: int=2020, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of players not participating in the fixtures for various reasons such as `suspended`, `injured` for example.
		Being a new endpoint, the data is only available from April 2021.
		
		**There are two types:**
		* `Missing Fixture` : The player will not play the fixture.
		* `Questionable` : The information is not yet 100% sure, the player may eventually play the fixture.
		
		> Examples available in Request samples "Use Cases".
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		**Update Frequency** : This endpoint is updated every 4 hours.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available injuries from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020`
		
		Get all available injuries from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?fixture=686314`
		
		Get all available injuries from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?team=85&season=2020`
		
		Get all available injuries from one {player} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?player=865&season=2020`
		
		Get all available injuries from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&player=865`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&timezone=Europe/London&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&league=61`"
    date: A valid date
        team: The id of the team
        player: The id of the player
        timezone: A valid timezone from the endpoint `Timezone`
        fixture: The id of the fixture
        league: The id of the league
        season: The season of the league, **required** with `league`, `team` and `player` parameters
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/injuries"
    querystring = {}
    if date:
        querystring['date'] = date
    if team:
        querystring['team'] = team
    if player:
        querystring['player'] = player
    if timezone:
        querystring['timezone'] = timezone
    if fixture:
        querystring['fixture'] = fixture
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_injuries_by_fixture_id(season: int=None, fixture: int=686308, date: str=None, league: int=None, team: int=None, timezone: str=None, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of players not participating in the fixtures for various reasons such as `suspended`, `injured` for example.
		Being a new endpoint, the data is only available from April 2021.
		
		**There are two types:**
		* `Missing Fixture` : The player will not play the fixture.
		* `Questionable` : The information is not yet 100% sure, the player may eventually play the fixture.
		
		> Examples available in Request samples "Use Cases".
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		**Update Frequency** : This endpoint is updated every 4 hours.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available injuries from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020`
		
		Get all available injuries from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?fixture=686314`
		
		Get all available injuries from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?team=85&season=2020`
		
		Get all available injuries from one {player} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?player=865&season=2020`
		
		Get all available injuries from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&player=865`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&timezone=Europe/London&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&league=61`"
    season: The season of the league, **required** with `league`, `team` and `player` parameters
        fixture: The id of the fixture
        date: A valid date
        league: The id of the league
        team: The id of the team
        timezone: A valid timezone from the endpoint `Timezone`
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/injuries"
    querystring = {}
    if season:
        querystring['season'] = season
    if fixture:
        querystring['fixture'] = fixture
    if date:
        querystring['date'] = date
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if timezone:
        querystring['timezone'] = timezone
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_injuries_by_date(season: int=None, player: int=None, league: int=None, date: str='2021-04-27', timezone: str=None, fixture: int=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of players not participating in the fixtures for various reasons such as `suspended`, `injured` for example.
		Being a new endpoint, the data is only available from April 2021.
		
		**There are two types:**
		* `Missing Fixture` : The player will not play the fixture.
		* `Questionable` : The information is not yet 100% sure, the player may eventually play the fixture.
		
		> Examples available in Request samples "Use Cases".
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		**Update Frequency** : This endpoint is updated every 4 hours.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available injuries from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020`
		
		Get all available injuries from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?fixture=686314`
		
		Get all available injuries from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?team=85&season=2020`
		
		Get all available injuries from one {player} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?player=865&season=2020`
		
		Get all available injuries from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?league=2&season=2020&player=865`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&timezone=Europe/London&team=85`
		`https://api-football-v1.p.rapidapi.com/v3/injuries?date=2021-04-07&league=61`"
    season: The season of the league, **required** with `league`, `team` and `player` parameters
        player: The id of the player
        league: The id of the league
        date: A valid date
        timezone: A valid timezone from the endpoint `Timezone`
        fixture: The id of the fixture
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/injuries"
    querystring = {}
    if season:
        querystring['season'] = season
    if player:
        querystring['player'] = player
    if league:
        querystring['league'] = league
    if date:
        querystring['date'] = date
    if timezone:
        querystring['timezone'] = timezone
    if fixture:
        querystring['fixture'] = fixture
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues(current: str=None, is_id: int=None, season: int=None, country: str=None, name: str=None, team: int=None, search: str=None, code: str=None, last: int=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    current: The state of the league
Enum: `true` or `false` 
        is_id: The id of the league
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        country: The country name of the league
        name: The name of the league
        team: The id of the team
        search: The name or the country of the league
`>= 3 characters`
        code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        last: The X last leagues/cups added in the API
`<= 2 characters`
        type: The type of the league 
 Enum: `league` or `cup` 
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if current:
        querystring['current'] = current
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if name:
        querystring['name'] = name
    if team:
        querystring['team'] = team
    if search:
        querystring['search'] = search
    if code:
        querystring['code'] = code
    if last:
        querystring['last'] = last
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_h2h_team_1_team_2(team_2: int, team_1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Head 2 Heads between 2 teams"
    team_2: team_id
        team_1: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/fixtures/h2h/{team_1}/{team_2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_league_id(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "!!! DEPRECIATED, USE "leagueTable endpoint" INSTEAD"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/standings/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_team_team_id(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one team"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/fixtures/team/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_league_league_id(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one league or cup"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/fixtures/league/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_live(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures in play"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/fixtures/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_date_date(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one date"
    date: YYYY-MM-DD
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/fixtures/date/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_league_league_id(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all teams from one league"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/teams/league/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_team_team_id(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one team from ID"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/teams/team/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_season_season(season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from season"
    season: YYYY
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/leagues/season/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_league_league_id(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one league from ID"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/leagues/league/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available leagues"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/leagues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get line ups from one fixture"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/lineups/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_top_assists(league: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 20 best players assists for a league or cup.
		
		**How it is calculated:**
		        * 1 : The player that has delivered the higher number of goal assists
		        * 2 : The player that has scored the higher number of goals
		        * 3 : The player that has scored the fewer number of penalties
		        * 4 : The player that assists in the higher number of matches
		        * 5 : The player that played the fewer minutes
		        * 6 : The player that received the fewer number of red cards
		        * 7 : The player that received the fewer number of yellow cards
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day."
    league: The id of the league
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players/topassists"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_top_red_cards(league: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 20 players with the most red cards for a league or cup.
		
		**How it is calculated:**
		        * 1 : The player that received the higher number of red cards
		        * 2 : The player that received the higher number of yellow cards
		        * 3 : The player that assists in the higher number of matches
		        * 4 : The player that played the fewer minutes
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day."
    league: The id of the league
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players/topredcards"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_available_from_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the events from a fixture
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the fixtures in progress otherwise 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/events/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rounds_available_for_a_league(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all rounds available from one league or cup, All spaces are replaced by underscore “_”
		**Update Frequency** : Every day.
		**Recommended Calls** : 1 call per day."
    league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/rounds/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coachs_from_team_id(coach_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the information about the coachs and their careers
		Update Frequency : Every day
		Recommended Calls : 1 call per day"
    coach_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/coachs/team/{coach_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_from_team_id(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available transfers for teams
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/transfers/team/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sidelined_from_player_id(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available sidelined for a player
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    player_id: player_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/sidelined/player/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sidelined_from_coach_id(coach_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available sidelined for a coach
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    coach_id: coach_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/sidelined/coach/{coach_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers
		All bookmakers `id` can be used in endpoint odds as filters
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/bookmakers/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_a_player(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to search for a player in relation to a player {name}"
    name: Player name
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/search/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_a_coach(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to search for a coach in relation to his name"
    keyword: keyWord
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/coachs/search/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_a_team(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to search for a team in relation to a {country} or team {name}. Spaces must be replaced by underscore. It is not necessary to put the {keyWord} in full, 3 characters are enough to search."
    keyword: keyWord
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/teams/search/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_a_league(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to search for a league in relation to a {country} or league {name}. Spaces must be replaced by underscore. It is not necessary to put the {keyWord} in full, 3 characters are enough to search."
    keyword: keyWord
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/search/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_available_for_players(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available seasons for players statistics
		The `season` returned by this endpoint are only available for `players endpoint`
		**Update Frequency** : This endpoint is updated every day
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics_filtered_by_season(season: str, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all statistics from one player filtered by season
		Statistics are calculated according to the `team_id`, `league_id` and `season`
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    season: season (YYYY or YYYY-YYYY)
        player_id: player_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/player/{player_id}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics_from_team(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all players statistics from one team
		Statistics are calculated according to the `team_id`, `league_id` and `season`
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/team/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all statistics from one player
		Statistics are calculated according to the `team_id`, `league_id` and `season`
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    player_id: player_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/player/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available countries"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_in_play_filtered_by_league_ids(league_id: str, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures in play filter by leagues ids (several leagues can be added by separating them with "-")
		In this request `events` are returned in the response
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    league_id: league_id-league_id-league_id etc...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/live/{league_id}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_from_team_id(team_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one team accross all seasons
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/team/{team_id}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_lineups_filtered_by_team_id(fixture: int, team: int=463, type: str=None, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the lineups for a fixture.
		
		**Available datas**
		        * Formation
		        * Coach
		        * Start XI
		        * Substitutes
		
		> Lineups are available between 20 and 40 minutes before the fixture.
		
		**Players' positions on the grid**
		
		**X** = row and **Y** = column (X:Y)
		
		Line 1 **X** being the one of the goal and then for each line this number is incremented. The column **Y** will go from left to right, and incremented for each player of the line.
		
		`As a new feature, some irregularities may occur, do not hesitate to report them on our public Roadmap`
		
		**Update Frequency** : This endpoint is updated every 15 minutes.
		**Recommended Calls** : 1 call every 15 minutes for the fixtures in progress otherwise 1 call per day.
		
		> Here are several examples of what can be done
		![demo-lineups](https://www.api-football.com/public/img/demo/demo-lineups-1.jpg)
		![demo-lineups](https://www.api-football.com/public/img/demo/demo-lineups.png)
		
		**Use Cases**
		Get all available lineups from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662`
		
		Get all available lineups from one {fixture} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&team=463`
		
		Get all available lineups from one {fixture} & {player}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&player=35845`
		
		Get all available lineups from one {fixture} & {type}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&type=startXI`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&player=35845&type=startXI`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups?fixture=215662&team=463&type=startXI&player=35845`"
    fixture: The id of the fixture
        team: The id of the team
        type: The Lineup type
Like `Formation`, `Substitutes`...
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_players_seasons_by_player_id(player: int=276, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available seasons for players statistics filtered by a player {id}.
		
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all seasons available for a player {id}
		`https://api-football-v1.p.rapidapi.com/v3/players/seasons?player=276`"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players/seasons"
    querystring = {}
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available timezone to be used in the fixtures endpoint.
		
		> This endpoint does not require any parameters.
		
		 **Update Frequency** : This endpoint contains all the existing timezone, it is not updated.
		**Recommended Calls** : 1 call when you need."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_from_team_id(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues in which the team has played at least one match"
    team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/team/{team_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_top_scorers(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the 20 best players for a league or cup.
		
		**How it is calculated:**
		        * 1 : The player that has scored the higher number of goals
		        * 2 : The player that has scored the fewer number of penalties
		        * 3 : The player that has delivered the higher number of goal assists
		        * 4 : The player that scored their goals in the higher number of matches
		        * 5 : The player that played the fewer minutes
		        * 6 : The player that plays for the team placed higher on the table
		        * 7 : The player that received the fewer number of red cards
		        * 8 : The player that received the fewer number of yellow cards
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day."
    season: The season of the league
`4 characters` Like 2020, 2021 ...
        league: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players/topscorers"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		The `name` and `code` fields can be used in other endpoints as filters."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_from_team_league_ids(league_id: int, team_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fixtures from one team filter by league
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    league_id: league_id
        team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/team/{team_id}/{league_id}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_from_league_id(league_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all teams from one league
		The `team id` are unique in the API and teams keep it among all the leagues/cups in which they participate"
    league_id: league_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/teams/league/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def h2h_between_2_teams(team_one: str, team_two: str, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Head 2 Heads between 2 teams
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/h2h/{team_one}/{team_two}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_from_id(fixture_id: int, timezone: str='Europe/London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one fixture from `ID`
		In this request `events`, `lineups`, `statistics` and `players statistics` are returned in the response
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    fixture_id: fixture_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/fixtures/id/{fixture_id}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_by_type_filtered_by_country_and_season(country: str, type: str, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve league by type (league or cup) and filtered by country and season"
    country: the country of the league
        type: the type of the league
        season: the season of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/type/{type}/{country}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_by_type_filtered_by_country(type: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve league by type (league or cup) and filtered by country"
    type: the type of the league
        country: the country of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/type/{type}/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_seasons_for_all_leagues_from_one_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve all the current seasons for all leagues and cups from one country"
    country: country
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/current/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_from_team_id_filtered_by_season(season: int, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues in which the team has played at least one match filter by season"
    season: season
        team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/team/{team_id}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available_seasons_for_a_league_filtered_by_season(league_id: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve all the seasons available in the API for a league/cup with the possibility to filter by {season}"
    league_id: league_id
        season: season - YYYY
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/seasonsAvailable/{league_id}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_from_season(season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues from season"
    season: YYYY
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/leagues/season/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_from_one_date(date: str, timezone: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one date
		you can filter by bookmaker or label like this :
		
		- /v2/odds/date/{yyyy-mm-dd}/bookmaker/{bookmakerid}
		- /v2/odds/date/{yyyy-mm-dd}/label/{labelid}
		
		**Update Frequency** : This endpoint is updated every day
		**Recommended Calls** : 1 call per day"
    timezone: Fails if field is not a result of the endpoint timezone
        page: Use for the pagination (Default : 1)
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/date/{date}"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_mapping(page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available odds
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    page: Use for the pagination (Default : 1)
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/mapping/"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_from_league_id(league_id: int, page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one league 
		you can filter by bookmaker or label like this : 
		- /v2/odds/league/{league_id}/bookmaker/{bookmaker_id}
		- /v2/odds/league/{league_id}/label/{label_id}
		
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    page: Use for the pagination (Default : 1)
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/league/{league_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_bets(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets.
		
		All bets `id` can be used in endpoint odds as filters.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available bets
		`https://api-football-v1.p.rapidapi.com/v3/odds/bets`
		
		Get bet from one {id}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bets?id=1`
		
		Allows you to search for a bet in relation to a bets {name}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bets?search=winner`"
    search: The name of the bet
        is_id: The id of the bet
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds/bets"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_by_league_id(date: str=None, page: int=None, bet: int=None, bookmaker: int=None, fixture: int=None, timezone: str=None, season: int=2020, league: int=287, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from fixtures, leagues or date.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages with to the `page` parameter.
		
		> **Pagination** : 10 results per page.
		
		We provide pre-match odds between 1 and 14 days before the fixture.
		
		We keep a 7-days history *(The availability of odds may vary according to the leagues, seasons, fixtures and bookmakers)*
		
		**Update Frequency** : This endpoint is updated every 3 hours.
		**Recommended Calls** : 1 call every 3 hours.
		
		**Use Cases**
		Get all available odds from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/odds?fixture=164327`
		
		Get all available odds from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/odds?league=39&season=2019`
		
		Get all available odds from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&bet=4&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bet=4&fixture=164327`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15&page=2&bet=4`"
    date: A valid date
        page: Use for the pagination
 Default: `1`
        bet: The id of the bet
        bookmaker: The id of the bookmaker
        fixture: The id of the fixture
        timezone: A valid timezone from the endpoint `Timezone`
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        league: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds"
    querystring = {}
    if date:
        querystring['date'] = date
    if page:
        querystring['page'] = page
    if bet:
        querystring['bet'] = bet
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if fixture:
        querystring['fixture'] = fixture
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_filtered_by_bet_id(league: int=None, bet: int=1, season: int=None, timezone: str=None, page: int=None, bookmaker: int=None, date: str=None, fixture: int=568987, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from fixtures, leagues or date.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages with to the `page` parameter.
		
		> **Pagination** : 10 results per page.
		
		We provide pre-match odds between 1 and 14 days before the fixture.
		
		We keep a 7-days history *(The availability of odds may vary according to the leagues, seasons, fixtures and bookmakers)*
		
		**Update Frequency** : This endpoint is updated every 3 hours.
		**Recommended Calls** : 1 call every 3 hours.
		
		**Use Cases**
		Get all available odds from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/odds?fixture=164327`
		
		Get all available odds from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/odds?league=39&season=2019`
		
		Get all available odds from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&bet=4&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bet=4&fixture=164327`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15&page=2&bet=4`"
    league: The id of the league
        bet: The id of the bet
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        timezone: A valid timezone from the endpoint `Timezone`
        page: Use for the pagination
 Default: `1`
        bookmaker: The id of the bookmaker
        date: A valid date
        fixture: The id of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds"
    querystring = {}
    if league:
        querystring['league'] = league
    if bet:
        querystring['bet'] = bet
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    if page:
        querystring['page'] = page
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if date:
        querystring['date'] = date
    if fixture:
        querystring['fixture'] = fixture
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_by_date(bet: int=None, bookmaker: int=None, page: int=None, timezone: str=None, date: str='2021-04-07', season: int=None, league: int=None, fixture: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from fixtures, leagues or date.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages with to the `page` parameter.
		
		> **Pagination** : 10 results per page.
		
		We provide pre-match odds between 1 and 14 days before the fixture.
		
		We keep a 7-days history *(The availability of odds may vary according to the leagues, seasons, fixtures and bookmakers)*
		
		**Update Frequency** : This endpoint is updated every 3 hours.
		**Recommended Calls** : 1 call every 3 hours.
		
		**Use Cases**
		Get all available odds from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/odds?fixture=164327`
		
		Get all available odds from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/odds?league=39&season=2019`
		
		Get all available odds from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&bet=4&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bet=4&fixture=164327`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15&page=2&bet=4`"
    bet: The id of the bet
        bookmaker: The id of the bookmaker
        page: Use for the pagination
 Default: `1`
        timezone: A valid timezone from the endpoint `Timezone`
        date: A valid date
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        league: The id of the league
        fixture: The id of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds"
    querystring = {}
    if bet:
        querystring['bet'] = bet
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if page:
        querystring['page'] = page
    if timezone:
        querystring['timezone'] = timezone
    if date:
        querystring['date'] = date
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    if fixture:
        querystring['fixture'] = fixture
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_from_fixture_id_filter_by_bookemaker(fixture_id: int, bookmaker_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one fixture 
		you can filter by bookmaker or label like this : 
		- /v2/odds/fixture/{fixture_id}/bookmaker/{bookmaker_id}
		- /v2/odds/fixture/{fixture_id}/label/{label_id}
		
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/fixture/{fixture_id}/bookmaker/{bookmaker_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_from_fixture_id_filter_by_label(fixture_id: int, label_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one fixture 
		you can filter by bookmaker or label like this : 
		- /v2/odds/fixture/{fixture_id}/bookmaker/{bookmaker_id}
		- /v2/odds/fixture/{fixture_id}/label/{label_id}
		
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/fixture/{fixture_id}/label/{label_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_from_league_id_filtered_by_label(label_id: str, league_id: int, page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available odds from one league 
		you can filter by bookmaker or label like this : 
		- /v2/odds/league/{league_id}/bookmaker/{bookmaker_id}
		- /v2/odds/league/{league_id}/label/{label_id}
		
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    page: Use for the pagination (Default : 1)
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/odds/league/{league_id}/label/{label_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_coachs_by_coach_id(is_id: int=276, team: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the information about the coachs and their careers.
		
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get coachs from one coach {id}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?id=1`
		
		Get coachs from one {team}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?team=33`
		
		Allows you to search for a coach in relation to a coach {name}
		`https://api-football-v1.p.rapidapi.com/v3/coachs?search=Klopp`"
    is_id: The id of the coach
        team: The id of the team
        search: The name of the coach
`>= 3 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/coachs"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_bookmakers(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers.
		
		All bookmakers `id` can be used in endpoint odds as filters.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available bookmakers
		`https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers`
		
		Get bookmaker from one {id}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers?id=1`
		
		Allows you to search for a bookmaker in relation to a bookmakers {name}
		`https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers?search=Betfair`"
    search: The name of the bookmaker
        is_id: The id of the bookmaker
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds/bookmakers"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_statistics(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all statistics from one fixture
		**Update Frequency** : Every  minute
		**Recommended Calls** : 1 call every minute for the fixtures in progress otherwise 1 call per day"
    fixture_id: fixture_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/statistics/fixture/{fixture_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_squad_from_team(season: str, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all players from one team for one season
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    season: season (YYYY or YYYY-YYYY)
        team_id: team_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/players/squad/{team_id}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_players_seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available seasons for players statistics.
		
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day."
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups_available_from_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lineups from one fixture
		**Update Frequency** : Every 15 minutes
		**Recommended Calls** : 1 call every 15 minutes for the fixtures in progress otherwise 1 call per day"
    fixture_id: fixture_id
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v2/lineups/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available seasons"
    
    """
    url = f"https://api-football-v1.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_predictions(fixture: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get predictions about a fixture.
		
		The predictions are made using several algorithms including the poisson distribution, comparison of team statistics, last matches, players etc…
		
		Bookmakers odds are not used to make these predictions
		
		Also provides some comparative statistics between teams
		
		**Available Predictions**
		          * Match winner : Id of the team that can potentially win the fixture
		          * Win or Draw : If `True` indicates that the designated team can win or draw
		          * Under / Over : -1.5 / -2.5 / -3.5 / -4.5 / +1.5 / +2.5 / +3.5 / +4.5 `*`
		          * Goals Home : -1.5 / -2.5 / -3.5 / -4.5 `*`
		          * Goals Away -1.5 / -2.5 / -3.5 / -4.5 `*`
		          * Advice *(Ex : Deportivo Santani or draws and -3.5 goals)*
		
		 `*` **-1.5** means that there will be a maximum of **1.5** goals in the fixture, i.e : **1** goal
		
		**Update Frequency** : This endpoint is updated every hour.
		**Recommended Calls** : 1 call per hour for the fixtures in progress otherwise 1 call per day.
		
		>Here is an example of what can be achieved
		![demo-prediction](https://www.api-football.com/public/img/demo/demo-prediction.png)
		
		**Use Cases**
		Get all available predictions from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/predictions?fixture=198772`"
    fixture: The id of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/predictions"
    querystring = {'fixture': fixture, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_transfers_by_team_id(player: int=None, team: int=33, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available transfers for players and teams
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all transfers from one {player}
		`https://api-football-v1.p.rapidapi.com/v3/transfers?player=35845`
		
		Get all transfers from one {team}
		`https://api-football-v1.p.rapidapi.com/v3/transfers?team=463`"
    player: The id of the player
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/transfers"
    querystring = {}
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_transfers_by_player_id(player: int=35845, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available transfers for players and teams
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all transfers from one {player}
		`https://api-football-v1.p.rapidapi.com/v3/transfers?player=35845`
		
		Get all transfers from one {team}
		`https://api-football-v1.p.rapidapi.com/v3/transfers?team=463`"
    player: The id of the player
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/transfers"
    querystring = {}
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_standings_by_league_id(season: int, league: int=39, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league or a team.
		
		Return a table of one or more rankings according to the league / cup.
		
		Some competitions have several rankings in a year, group phase, opening ranking, closing ranking etc…
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated every hour.
		**Recommended Calls** : 1 call per hour for the leagues or teams who have at least one fixture in progress otherwise 1 call per day.
		
		**Use Cases**
		Get all Standings from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/standings?league=39&season=2019`
		
		Get all Standings from one {league} & {season} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/standings?league=39&team=33&season=2019`
		
		Get all Standings from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/standings?team=33&season=2019`"
    season: The season of the league
`4 characters` Like 2020, 2021 ...
        league: The id of the league
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/standings"
    querystring = {'season': season, }
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_standings_by_team_id(season: int, league: int=None, team: int=33, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league or a team.
		
		Return a table of one or more rankings according to the league / cup.
		
		Some competitions have several rankings in a year, group phase, opening ranking, closing ranking etc…
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated every hour.
		**Recommended Calls** : 1 call per hour for the leagues or teams who have at least one fixture in progress otherwise 1 call per day.
		
		**Use Cases**
		Get all Standings from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/standings?league=39&season=2019`
		
		Get all Standings from one {league} & {season} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/standings?league=39&team=33&season=2019`
		
		Get all Standings from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/standings?team=33&season=2019`"
    season: The season of the league
`4 characters` Like 2020, 2021 ...
        league: The id of the league
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/standings"
    querystring = {'season': season, }
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_trophies_by_player_id(coach: int=None, player: int=276, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available trophies for a player or a coach.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all trophies from one {player}
		`https://api-football-v1.p.rapidapi.com/v3/trophies?player=276`
		
		Get all trophies from one {coach}
		`https://api-football-v1.p.rapidapi.com/v3/trophies?coach=2`"
    coach: The id of the coach
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/trophies"
    querystring = {}
    if coach:
        querystring['coach'] = coach
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_sidelined_by_coach_id(coach: int=276, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available sidelined for a player or a coach.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all from one {player}
		`https://api-football-v1.p.rapidapi.com/v3/sidelined?player=276`
		
		Get all from one {coach}
		`https://api-football-v1.p.rapidapi.com/v3/sidelined?coach=2`"
    coach: The id of the coach
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/sidelined"
    querystring = {}
    if coach:
        querystring['coach'] = coach
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_trophies_by_coach_id(coach: int=276, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available trophies for a player or a coach.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all trophies from one {player}
		`https://api-football-v1.p.rapidapi.com/v3/trophies?player=276`
		
		Get all trophies from one {coach}
		`https://api-football-v1.p.rapidapi.com/v3/trophies?coach=2`"
    coach: The id of the coach
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/trophies"
    querystring = {}
    if coach:
        querystring['coach'] = coach
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_players_statistics_by_league_id(page: int=None, search: str=None, league: int=39, team: int=None, is_id: int=None, season: int=2020, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players statistics.
		
		The players `id` are unique in the API and players keep it among all the teams they have been in
		
		The statistics are calculated according to the team `id`, league `id` and `season`.
		
		You can find the available `seasons` by using the endpoint players seasons.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages thanks to the `page` parameter.
		
		> **Pagination** : 20 results per page.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all players statistics from one player {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?id=19088&season=2018`
		
		Get all players statistics from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33&page=2`
		
		Get all players statistics from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&page=4`
		
		Get all players statistics from one {league}, {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33&page=5`
		
		Allows you to search for a player in relation to a player {name}
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?league=61&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani&season=2018`"
    page: Use for the pagination
Default: `1`
        search: The name of the player `>= 4 characters`
Requires the fields `League` or `Team`
        league: The id of the league
        team: The id of the team
        is_id: The id of the player
        season: The season of the league `4 characters`
Requires the fields `Id`, `League` or `Team`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players"
    querystring = {}
    if page:
        querystring['page'] = page
    if search:
        querystring['search'] = search
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_players_statistics_by_team_id(season: int=2020, team: int=33, league: int=None, search: str=None, page: int=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players statistics.
		
		The players `id` are unique in the API and players keep it among all the teams they have been in
		
		The statistics are calculated according to the team `id`, league `id` and `season`.
		
		You can find the available `seasons` by using the endpoint players seasons.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages thanks to the `page` parameter.
		
		> **Pagination** : 20 results per page.
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all players statistics from one player {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?id=19088&season=2018`
		
		Get all players statistics from one {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&team=33&page=2`
		
		Get all players statistics from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&page=4`
		
		Get all players statistics from one {league}, {team} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33`
		`https://api-football-v1.p.rapidapi.com/v3/players?season=2018&league=61&team=33&page=5`
		
		Allows you to search for a player in relation to a player {name}
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?league=61&search=cavani`
		`https://api-football-v1.p.rapidapi.com/v3/players?team=85&search=cavani&season=2018`"
    season: The season of the league `4 characters`
Requires the fields `Id`, `League` or `Team`
        team: The id of the team
        league: The id of the league
        search: The name of the player `>= 4 characters`
Requires the fields `League` or `Team`
        page: Use for the pagination
Default: `1`
        is_id: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/players"
    querystring = {}
    if season:
        querystring['season'] = season
    if team:
        querystring['team'] = team
    if league:
        querystring['league'] = league
    if search:
        querystring['search'] = search
    if page:
        querystring['page'] = page
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_events(fixture: int, team: int=None, type: str=None, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the events from a fixture.
		
		**Available events**   
		        * Goal  : Normal Goal, Own Goal, Penalty, Missed Penalty
		        * Card : Yellow Card, Second Yellow card, Red card
		        * Subst : Substitution [1, 2, 3...]
		        * Var : Goal cancelled, Penalty confirmed
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the fixtures in progress otherwise 1 call per day.
		
		You can also retrieve all the events of the fixtures in progress thanks to the endpoint `fixtures?live=all`
		
		> Here is an example of what can be achieved
		![demo-events](https://www.api-football.com/public/img/demo/demo-events.png)
		
		**Use Cases**
		Get all available events from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662`
		
		Get all available events from one {fixture} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&team=463`
		
		Get all available events from one {fixture} & {player}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&player=35845`
		
		Get all available events from one {fixture} & {type}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&type=card`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&player=35845&type=card`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&team=463&type=goal&player=35845`"
    fixture: The id of the fixture
        team: The id of the team
        type: The event type
Like `Goal`, `Card` ...
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/events"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_events_filtered_by_player_id(fixture: int, team: int=None, type: str=None, player: int=6126, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the events from a fixture.
		
		**Available events**   
		        * Goal  : Normal Goal, Own Goal, Penalty, Missed Penalty
		        * Card : Yellow Card, Second Yellow card, Red card
		        * Subst : Substitution [1, 2, 3...]
		        * Var : Goal cancelled, Penalty confirmed
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the fixtures in progress otherwise 1 call per day.
		
		You can also retrieve all the events of the fixtures in progress thanks to the endpoint `fixtures?live=all`
		
		> Here is an example of what can be achieved
		![demo-events](https://www.api-football.com/public/img/demo/demo-events.png)
		
		**Use Cases**
		Get all available events from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662`
		
		Get all available events from one {fixture} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&team=463`
		
		Get all available events from one {fixture} & {player}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&player=35845`
		
		Get all available events from one {fixture} & {type}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&type=card`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&player=35845&type=card`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&team=463&type=goal&player=35845`"
    fixture: The id of the fixture
        team: The id of the team
        type: The event type
Like `Goal`, `Card` ...
        player: The id of the player
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/events"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_events_filtered_by_team_id(fixture: int, type: str=None, player: int=None, team: int=463, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the events from a fixture.
		
		**Available events**   
		        * Goal  : Normal Goal, Own Goal, Penalty, Missed Penalty
		        * Card : Yellow Card, Second Yellow card, Red card
		        * Subst : Substitution [1, 2, 3...]
		        * Var : Goal cancelled, Penalty confirmed
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the fixtures in progress otherwise 1 call per day.
		
		You can also retrieve all the events of the fixtures in progress thanks to the endpoint `fixtures?live=all`
		
		> Here is an example of what can be achieved
		![demo-events](https://www.api-football.com/public/img/demo/demo-events.png)
		
		**Use Cases**
		Get all available events from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662`
		
		Get all available events from one {fixture} & {team}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&team=463`
		
		Get all available events from one {fixture} & {player}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&player=35845`
		
		Get all available events from one {fixture} & {type}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&type=card`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&player=35845&type=card`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures/events?fixture=215662&team=463&type=goal&player=35845`"
    fixture: The id of the fixture
        type: The event type
Like `Goal`, `Card` ...
        player: The id of the player
        team: The id of the team
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/events"
    querystring = {'fixture': fixture, }
    if type:
        querystring['type'] = type
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_venues_by_country_name(name: str=None, is_id: int=None, country: str='England', city: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available venues.
		
		The venue `id` are **unique** in the API.
		
		> All the parameters of this endpoint can be used together.
		
		**This endpoint requires at least one parameter.**
		
		**Update Frequency** : This endpoint is updated several times a week.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get one venue from venue {id}
		`https://api-football-v1.p.rapidapi.com/v3/venues?id=556`
		
		Get one venue from venue {name}
		`https://api-football-v1.p.rapidapi.com/v3/venues?name=Old Trafford`
		
		Get all venues from {city}
		`https://api-football-v1.p.rapidapi.com/v3/venues?city=manchester`
		
		Get venues from {country}
		`https://api-football-v1.p.rapidapi.com/v3/venues?country=england`
		
		Allows you to search for a venues in relation to a venue {name}, {city} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=trafford`
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=manches`
		`https://api-football-v1.p.rapidapi.com/v3/venues?search=England`"
    name: The name of the venue
        is_id: The id of the venue
        country: The country name of the venue
        city: The city of the venue
        search: The name, city or the country of the venue
`>= 3 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/venues"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    if city:
        querystring['city'] = city
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_statistics_filtered_by_team_id(fixture: int, team: int=463, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics for one fixture.
		
		**Available statistics**
		        * Shots on Goal
		        * Shots off Goal
		        * Shots insidebox
		        * Shots outsidebox
		        * Total Shots
		        * Blocked Shots
		        * Fouls
		        * Corner Kicks
		        * Offsides
		        * Ball Possession
		        * Yellow Cards
		        * Red Cards
		        * Goalkeeper Saves
		        * Total passes
		        * Passes accurate
		        * Passes %
		
		**Update Frequency** : This endpoint is updated every minute.
		**Recommended Calls** : 1 call every minute for the teams or fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here is an example of what can be achieved
		![demo-statistics](https://www.api-football.com/public/img/demo/demo-statistics.png)
		
		**Use Cases**
		Get all available statistics from one {fixture}
		`https://v3.football.api-sports.io/fixtures/statistics?fixture=215662`
		
		Get all available statistics from one {fixture} & {type}
		`https://v3.football.api-sports.io/fixtures/statistics?fixture=215662&type=Total Shots`
		
		Get all available statistics from one {fixture} & {team}
		`v3.football.api-sports.io/fixtures/statistics?fixture=215662&team=463`"
    fixture: The id of the fixture
        team: The id of the team
        type: The type of statistics
Like `Fouls`, `Offsides`...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_statistics_filtered_by_type(fixture: int, team: int=None, type: str='Shots on Goal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics for one fixture.
		
		**Available statistics**
		        * Shots on Goal
		        * Shots off Goal
		        * Shots insidebox
		        * Shots outsidebox
		        * Total Shots
		        * Blocked Shots
		        * Fouls
		        * Corner Kicks
		        * Offsides
		        * Ball Possession
		        * Yellow Cards
		        * Red Cards
		        * Goalkeeper Saves
		        * Total passes
		        * Passes accurate
		        * Passes %
		
		**Update Frequency** : This endpoint is updated every minute.
		**Recommended Calls** : 1 call every minute for the teams or fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here is an example of what can be achieved
		![demo-statistics](https://www.api-football.com/public/img/demo/demo-statistics.png)
		
		**Use Cases**
		Get all available statistics from one {fixture}
		`https://v3.football.api-sports.io/fixtures/statistics?fixture=215662`
		
		Get all available statistics from one {fixture} & {type}
		`https://v3.football.api-sports.io/fixtures/statistics?fixture=215662&type=Total Shots`
		
		Get all available statistics from one {fixture} & {team}
		`v3.football.api-sports.io/fixtures/statistics?fixture=215662&team=463`"
    fixture: The id of the fixture
        team: The id of the team
        type: The type of statistics
Like `Fouls`, `Offsides`...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_by_fixture_id(date: str=None, fixture: int=568987, page: int=None, league: int=None, bet: int=None, season: int=None, timezone: str=None, bookmaker: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from fixtures, leagues or date.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages with to the `page` parameter.
		
		> **Pagination** : 10 results per page.
		
		We provide pre-match odds between 1 and 14 days before the fixture.
		
		We keep a 7-days history *(The availability of odds may vary according to the leagues, seasons, fixtures and bookmakers)*
		
		**Update Frequency** : This endpoint is updated every 3 hours.
		**Recommended Calls** : 1 call every 3 hours.
		
		**Use Cases**
		Get all available odds from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/odds?fixture=164327`
		
		Get all available odds from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/odds?league=39&season=2019`
		
		Get all available odds from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&bet=4&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bet=4&fixture=164327`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15&page=2&bet=4`"
    date: A valid date
        fixture: The id of the fixture
        page: Use for the pagination
 Default: `1`
        league: The id of the league
        bet: The id of the bet
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        timezone: A valid timezone from the endpoint `Timezone`
        bookmaker: The id of the bookmaker
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds"
    querystring = {}
    if date:
        querystring['date'] = date
    if fixture:
        querystring['fixture'] = fixture
    if page:
        querystring['page'] = page
    if league:
        querystring['league'] = league
    if bet:
        querystring['bet'] = bet
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_odds_filtered_by_bookmaker_id(bet: int=None, bookmaker: int=1, date: str=None, fixture: int=568987, league: int=None, page: int=None, timezone: str=None, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from fixtures, leagues or date.
		
		This endpoint uses a **pagination system**, you can navigate between the different pages with to the `page` parameter.
		
		> **Pagination** : 10 results per page.
		
		We provide pre-match odds between 1 and 14 days before the fixture.
		
		We keep a 7-days history *(The availability of odds may vary according to the leagues, seasons, fixtures and bookmakers)*
		
		**Update Frequency** : This endpoint is updated every 3 hours.
		**Recommended Calls** : 1 call every 3 hours.
		
		**Use Cases**
		Get all available odds from one {fixture}
		`https://api-football-v1.p.rapidapi.com/v3/odds?fixture=164327`
		
		Get all available odds from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/odds?league=39&season=2019`
		
		Get all available odds from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&bet=4&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bet=4&fixture=164327`
		`https://api-football-v1.p.rapidapi.com/v3/odds?bookmaker=1&league=39&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/odds?date=2020-05-15&page=2&bet=4`"
    bet: The id of the bet
        bookmaker: The id of the bookmaker
        date: A valid date
        fixture: The id of the fixture
        league: The id of the league
        page: Use for the pagination
 Default: `1`
        timezone: A valid timezone from the endpoint `Timezone`
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds"
    querystring = {}
    if bet:
        querystring['bet'] = bet
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if date:
        querystring['date'] = date
    if fixture:
        querystring['fixture'] = fixture
    if league:
        querystring['league'] = league
    if page:
        querystring['page'] = page
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_teams_statistics(league: int, season: int, team: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Update Frequency** : This endpoint is updated twice a day.
		**Recommended Calls** : 1 call per day for the teams who have at least one fixture during the day otherwise 1 call per week.
		
		> Here is an example of what can be achieved
		![demo-teams-statistics](https://www.api-football.com/public/img/demo/demo-teams-statistics.png)
		
		**Use Cases**
		Get all statistics for a {team} in a {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/teams/statistics?league=39&team=33&season=2019`
		
		Get all statistics for a {team} in a {league} & {season} with a end {date}
		`https://api-football-v1.p.rapidapi.com/v3/teams/statistics?league=39&team=33&season=2019&date=2019-10-08`"
    league: The id of the league
        season: The season of the league
` 4 characters`Like 2020, 2021 ...
        team: The id of the team
        date: The limit date
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics"
    querystring = {'league': league, 'season': season, 'team': team, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues_by_league_id(name: str=None, is_id: int=39, season: int=None, country: str=None, team: int=None, code: str=None, type: str=None, search: str=None, current: str=None, last: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    name: The name of the league
        is_id: The id of the league
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        country: The country name of the league
        team: The id of the team
        code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        type: The type of the league 
 Enum: `league` or `cup` 
        search: The name or the country of the league
`>= 3 characters`
        current: The state of the league
Enum: `true` or `false` 
        last: The X last leagues/cups added in the API
`<= 2 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if team:
        querystring['team'] = team
    if code:
        querystring['code'] = code
    if type:
        querystring['type'] = type
    if search:
        querystring['search'] = search
    if current:
        querystring['current'] = current
    if last:
        querystring['last'] = last
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues_by_country_name(search: str=None, name: str=None, code: str=None, is_id: int=None, team: int=None, current: str=None, season: int=None, country: str='England', type: str=None, last: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    search: The name or the country of the league
`>= 3 characters`
        name: The name of the league
        code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        is_id: The id of the league
        team: The id of the team
        current: The state of the league
Enum: `true` or `false` 
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        country: The country name of the league
        type: The type of the league 
 Enum: `league` or `cup` 
        last: The X last leagues/cups added in the API
`<= 2 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if code:
        querystring['code'] = code
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    if current:
        querystring['current'] = current
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if type:
        querystring['type'] = type
    if last:
        querystring['last'] = last
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues_by_team_id(code: str=None, name: str=None, type: str=None, is_id: int=None, team: int=33, current: str=None, last: int=None, search: str=None, country: str=None, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        name: The name of the league
        type: The type of the league 
 Enum: `league` or `cup` 
        is_id: The id of the league
        team: The id of the team
        current: The state of the league
Enum: `true` or `false` 
        last: The X last leagues/cups added in the API
`<= 2 characters`
        search: The name or the country of the league
`>= 3 characters`
        country: The country name of the league
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if code:
        querystring['code'] = code
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    if current:
        querystring['current'] = current
    if last:
        querystring['last'] = last
    if search:
        querystring['search'] = search
    if country:
        querystring['country'] = country
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues_by_type_league_or_cup(type: str='cup', name: str=None, is_id: int=None, team: int=None, country: str=None, last: int=None, search: str=None, current: str=None, season: int=None, code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    type: The type of the league 
 Enum: `league` or `cup` 
        name: The name of the league
        is_id: The id of the league
        team: The id of the team
        country: The country name of the league
        last: The X last leagues/cups added in the API
`<= 2 characters`
        search: The name or the country of the league
`>= 3 characters`
        current: The state of the league
Enum: `true` or `false` 
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if type:
        querystring['type'] = type
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    if country:
        querystring['country'] = country
    if last:
        querystring['last'] = last
    if search:
        querystring['search'] = search
    if current:
        querystring['current'] = current
    if season:
        querystring['season'] = season
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues_by_country_code(current: str=None, season: int=None, name: str=None, type: str=None, country: str=None, team: int=None, search: str=None, last: int=None, code: str='it', is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    current: The state of the league
Enum: `true` or `false` 
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        name: The name of the league
        type: The type of the league 
 Enum: `league` or `cup` 
        country: The country name of the league
        team: The id of the team
        search: The name or the country of the league
`>= 3 characters`
        last: The X last leagues/cups added in the API
`<= 2 characters`
        code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        is_id: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if current:
        querystring['current'] = current
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    if country:
        querystring['country'] = country
    if team:
        querystring['team'] = team
    if search:
        querystring['search'] = search
    if last:
        querystring['last'] = last
    if code:
        querystring['code'] = code
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues_by_season(country: str=None, team: int=None, current: str=None, season: int=2020, type: str=None, name: str=None, code: str=None, search: str=None, is_id: int=None, last: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    country: The country name of the league
        team: The id of the team
        current: The state of the league
Enum: `true` or `false` 
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        type: The type of the league 
 Enum: `league` or `cup` 
        name: The name of the league
        code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        search: The name or the country of the league
`>= 3 characters`
        is_id: The id of the league
        last: The X last leagues/cups added in the API
`<= 2 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if country:
        querystring['country'] = country
    if team:
        querystring['team'] = team
    if current:
        querystring['current'] = current
    if season:
        querystring['season'] = season
    if type:
        querystring['type'] = type
    if name:
        querystring['name'] = name
    if code:
        querystring['code'] = code
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    if last:
        querystring['last'] = last
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_leagues_whose_season_is_running(code: str=None, current: str='true', search: str=None, type: str=None, country: str=None, team: int=None, last: int=None, name: str=None, season: int=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together.
		
		**Update Frequency** : This endpoint is updated several times a day.
		**Recommended Calls** : 1 call per hour.
		
		**Use Cases**
		Allows to retrieve all the seasons available for a league/cup
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=39`
		
		Get all leagues from one league {name}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?name=premier league`
		
		Get all leagues from one {country}
		You can find the available {country} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?country=england`
		
		Get all leagues from one country {code} (GB, FR, IT etc..)
		You can find the available country {code} by using the endpoint country
		`https://api-football-v1.p.rapidapi.com/v3/leagues?code=gb`
		
		Get all leagues from one {season}
		You can find the available {season} by using the endpoint seasons
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019`
		
		Get one league from one league {id} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&id=39`
		
		Get all leagues in which the {team} has played at least one match
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=33`
		
		Allows you to search for a league in relation to a league {name} or {country}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=premier league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?search=England`
		
		Get all leagues from one {type}
		`https://api-football-v1.p.rapidapi.com/v3/leagues?type=league`
		
		Get all leagues where the season is in progress or not
		`https://api-football-v1.p.rapidapi.com/v3/leagues?current=true`
		
		Get the last 99 leagues or cups added to the API
		`https://api-football-v1.p.rapidapi.com/v3/leagues?last=99`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/leagues?season=2019&country=england&type=league`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?team=85&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/leagues?id=61&current=true&type=league`"
    code: The Alpha2 code of the country
`2 characters` Like FR, GB, IT…
        current: The state of the league
Enum: `true` or `false` 
        search: The name or the country of the league
`>= 3 characters`
        type: The type of the league 
 Enum: `league` or `cup` 
        country: The country name of the league
        team: The id of the team
        last: The X last leagues/cups added in the API
`<= 2 characters`
        name: The name of the league
        season: The season of the league
`4 characters` Like 2018, 2019 etc...
        is_id: The id of the league
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/leagues"
    querystring = {}
    if code:
        querystring['code'] = code
    if current:
        querystring['current'] = current
    if search:
        querystring['search'] = search
    if type:
        querystring['type'] = type
    if country:
        querystring['country'] = country
    if team:
        querystring['team'] = team
    if last:
        querystring['last'] = last
    if name:
        querystring['name'] = name
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_rounds(league: int, season: int, current: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the rounds for a league or a cup.
		
		The `round` can be used in endpoint fixtures as filters
		   
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available rounds from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/rounds?league=39&season=2019`
		
		Get current round from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/rounds?league=39&season=2019&current=true`"
    league: The id of the league
        season: The season of the league
`4 characters` Like 2020, 2021 ...
        current: Retrun the current round only
 Enum: `true` or `false` 
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/rounds"
    querystring = {'league': league, 'season': season, }
    if current:
        querystring['current'] = current
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_current_round(season: int, league: int, current: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the rounds for a league or a cup.
		
		The `round` can be used in endpoint fixtures as filters
		   
		**Update Frequency** : This endpoint is updated every day.
		**Recommended Calls** : 1 call per day.
		
		**Use Cases**
		Get all available rounds from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/rounds?league=39&season=2019`
		
		Get current round from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/rounds?league=39&season=2019&current=true`"
    season: The season of the league
`4 characters` Like 2020, 2021 ...
        league: The id of the league
        current: Retrun the current round only
 Enum: `true` or `false` 
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures/rounds"
    querystring = {'season': season, 'league': league, }
    if current:
        querystring['current'] = current
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_by_date(live: str=None, ids: str=None, league: int=None, to: str=None, date: str='2021-04-07', venue: int=None, timezone: str=None, status: str=None, season: int=None, is_from: str=None, team: int=None, last: int=None, round: str=None, is_id: int=None, next: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    live:  Enum: `all` or `id-id` for filter by league id 
        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        league: The id of the league
        to: A valid date
        date: A valid date
        venue: The venue id of the fixture
        timezone: A valid timezone from the endpoint `Timezone`
        status: One or more fixture status short
Like NS or NS-FT-CANC
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        is_from: A valid date

        team: The id of the team
        last: For the X last fixtures
`<= 2 characters`
        round: The round of the fixture
        is_id: The id of the fixture
        next: For the X next fixtures
`<= 2 characters`
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if live:
        querystring['live'] = live
    if ids:
        querystring['ids'] = ids
    if league:
        querystring['league'] = league
    if to:
        querystring['to'] = to
    if date:
        querystring['date'] = date
    if venue:
        querystring['venue'] = venue
    if timezone:
        querystring['timezone'] = timezone
    if status:
        querystring['status'] = status
    if season:
        querystring['season'] = season
    if is_from:
        querystring['from'] = is_from
    if team:
        querystring['team'] = team
    if last:
        querystring['last'] = last
    if round:
        querystring['round'] = round
    if is_id:
        querystring['id'] = is_id
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_fixtures_by_team_id(season: int=2020, to: str=None, venue: int=None, ids: str=None, team: int=33, timezone: str=None, is_id: int=None, date: str=None, league: int=None, live: str=None, last: int=None, next: int=None, is_from: str=None, status: str=None, round: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    season: The season of the league
`4 characters` Like 2020; 2021 ...
        to: A valid date
        venue: The venue id of the fixture
        ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        team: The id of the team
        timezone: A valid timezone from the endpoint `Timezone`
        is_id: The id of the fixture
        date: A valid date
        league: The id of the league
        live:  Enum: `all` or `id-id` for filter by league id 
        last: For the X last fixtures
`<= 2 characters`
        next: For the X next fixtures
`<= 2 characters`
        is_from: A valid date

        status: One or more fixture status short
Like NS or NS-FT-CANC
        round: The round of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if season:
        querystring['season'] = season
    if to:
        querystring['to'] = to
    if venue:
        querystring['venue'] = venue
    if ids:
        querystring['ids'] = ids
    if team:
        querystring['team'] = team
    if timezone:
        querystring['timezone'] = timezone
    if is_id:
        querystring['id'] = is_id
    if date:
        querystring['date'] = date
    if league:
        querystring['league'] = league
    if live:
        querystring['live'] = live
    if last:
        querystring['last'] = last
    if next:
        querystring['next'] = next
    if is_from:
        querystring['from'] = is_from
    if status:
        querystring['status'] = status
    if round:
        querystring['round'] = round
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_last_x_fixtures_that_were_played(ids: str=None, status: str=None, venue: int=None, is_from: str=None, is_id: int=None, timezone: str=None, season: int=None, to: str=None, date: str=None, league: int=None, live: str=None, next: int=None, last: int=50, team: int=None, round: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to fixtures you can add the query parameter `timezone` to your request in order to retrieve the list of matches in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		> Some leagues have only final result check our coverage page to know which ones
		
		**Available fixtures status**
		        * TBD : Time To Be Defined
		        * NS : Not Started
		        * 1H : First Half, Kick Off
		        * HT : Halftime
		        * 2H : Second Half, 2nd Half Started
		        * ET : Extra Time
		        * P : Penalty In Progress
		        * FT : Match Finished
		        * AET : Match Finished After Extra Time
		        * PEN : Match Finished After Penalty
		        * BT : Break Time (in Extra Time)
		        * SUSP : Match Suspended
		        * INT : Match Interrupted
		        * PST : Match Postponed
		        * CANC : Match Cancelled
		        * ABD : Match Abandoned
		        * AWD : Technical Loss
		        * WO : WalkOver
		
		**Update Frequency** : This endpoint is updated every 15 seconds.
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day.
		
		> Here are several examples of what can be achieved
		![demo-fixtures](https://www.api-football.com/public/img/demo/demo-fixtures.jpg)
		
		**Use Cases**
		Get fixture from one fixture {id}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?id=215662`
		
		Get fixture from severals fixtures {ids}
		In this request events, lineups, statistics fixture and players fixture are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?ids=215662-215663-215664-215665-215666-215667`
		
		Get all available fixtures in play
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all`
		
		Get all available fixtures in play filter by several {league}
		In this request events are returned in the response
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?live=39-61-48`
		
		Get all available fixtures from one {league} & {season}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=39&season=2019`
		
		Get all available fixtures from one {date}
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2019-10-22`
		
		Get next X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?next=15`
		
		Get last X available fixtures
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?last=15`
		
		It’s possible to make requests by mixing the available parameters
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?date=2020-01-30&league=61&season=2019`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&next=10`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&last=10&status=ft`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&last=10&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?team=85&season=2019&from=2019-07-01&to=2020-10-31`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&from=2019-07-01&to=2020-10-31&timezone=Europe/london`
		`https://api-football-v1.p.rapidapi.com/v3/fixtures?league=61&season=2019&round=Regular Season - 1`"
    ids: One or more fixture ids
`id-id-id`
Maximum of 20 fixtures ids
        status: One or more fixture status short
Like NS or NS-FT-CANC
        venue: The venue id of the fixture
        is_from: A valid date

        is_id: The id of the fixture
        timezone: A valid timezone from the endpoint `Timezone`
        season: The season of the league
`4 characters` Like 2020; 2021 ...
        to: A valid date
        date: A valid date
        league: The id of the league
        live:  Enum: `all` or `id-id` for filter by league id 
        next: For the X next fixtures
`<= 2 characters`
        last: For the X last fixtures
`<= 2 characters`
        team: The id of the team
        round: The round of the fixture
        
    """
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if status:
        querystring['status'] = status
    if venue:
        querystring['venue'] = venue
    if is_from:
        querystring['from'] = is_from
    if is_id:
        querystring['id'] = is_id
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    if to:
        querystring['to'] = to
    if date:
        querystring['date'] = date
    if league:
        querystring['league'] = league
    if live:
        querystring['live'] = live
    if next:
        querystring['next'] = next
    if last:
        querystring['last'] = last
    if team:
        querystring['team'] = team
    if round:
        querystring['round'] = round
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

