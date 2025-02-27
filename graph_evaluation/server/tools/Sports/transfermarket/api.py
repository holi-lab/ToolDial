import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def transfers_list_rumors(competitionids: str, maxvalue: int=None, minvalue: int=None, clubids: str=None, positiongroup: str=None, secodarypositions: bool=None, playerids: str=None, positionid: int=None, hideclosed: bool=None, sort: str='date_desc', domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List transfer rumors along with players info, and estimated market value"
    competitionids: The value of id field returned in .../search or .../competitions/list... endpoints. DO NOT use together with clubIds and  playerIds parameters. Separated by coma for multiple values. Ex : IT1,GB1,etc...
        maxvalue: The max market value of players. Ex : 151000000
        minvalue: The min market value of players. Ex : 9000000
        clubids: The value of id field returned in .../search or .../clubs/list-by-competition endpoints. DO NOT use together with competitionIds and playerIds parameters. Separated by coma for multiple values. Ex : 631,383,985,27,etc...
        positiongroup: One of the following Abwehr|Mittelfeld|Sturm
        secodarypositions: true|false
        playerids: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. DO NOT use together with clubIds and competitionIds parameter. Separated by coma for multiple values. Ex : 68778,405686,etc...
        positionid: One of the following 1-Goalkeeper|2-Sweeper|3-Centre-Back|4-Left-Back|5-Right-Back|6-Defensive Midfield|7-Central Midfield|8-Right Midfield|9-Left Midfield|10-Attacking Midfield|11-Left Winger|12-Right Winger|13-Second Striker|14-Centre-Forward
        hideclosed: true|false
        sort: One of the following date_desc|probability_desc
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/transfers/list-rumors"
    querystring = {'competitionIds': competitionids, }
    if maxvalue:
        querystring['maxValue'] = maxvalue
    if minvalue:
        querystring['minValue'] = minvalue
    if clubids:
        querystring['clubIds'] = clubids
    if positiongroup:
        querystring['positionGroup'] = positiongroup
    if secodarypositions:
        querystring['secodaryPositions'] = secodarypositions
    if playerids:
        querystring['playerIds'] = playerids
    if positionid:
        querystring['positionId'] = positionid
    if hideclosed:
        querystring['hideClosed'] = hideclosed
    if sort:
        querystring['sort'] = sort
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_get_table(seasonid: str, is_id: str, homeaway: str=None, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get table by competition and season"
    seasonid: The value of id field returned in .../competitions/list-seasons endpoint
        id: The value of id field returned in .../search or .../competitions/list-default endpoints
        homeaway: One of the following : home|away
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/competitions/get-table"
    querystring = {'seasonID': seasonid, 'id': is_id, }
    if homeaway:
        querystring['homeAway'] = homeaway
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_header_info(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get header information of player on the top of page"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. 
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-header-info"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referees_get_profile(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get profile of specific referee"
    id: The value of id field returned in .../search endpoint
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/referees/get-profile"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_competition(is_id: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news relating to specific competition or league"
    id: The value of id field returned in .../search or .../competitions/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/news/list-by-competition"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_game_information(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game information of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-game-information"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_stats(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-stats"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list_by_club(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List matches by specific club"
    id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/list-by-club"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list_by_game_plan(dayid: int, seasonid: int, leagueid: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List matches by game plan"
    dayid: The value of gamePlanPlayDays/id JSON object returned in .../competitions/get-game-plan endpoint.
        seasonid: The value of id field returned in .../competitions/list-seasons endpoint
        leagueid: The value of id field returned in .../search or .../competitions/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/list-by-game-plan"
    querystring = {'dayID': dayid, 'seasonID': seasonid, 'leagueID': leagueid, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_uefa_5year_rankings(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List UEFA 5-year rankings"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-uefa-5year-rankings"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_get_header_info(is_id: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief description of specific competition"
    id: The value of id field returned in .../search or .../competitions/list-default endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/competitions/get-header-info"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_list_seasons(is_id: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List passed seasons of specific competition"
    id: The value of id field returned in .../search or .../competitions/list-default endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/competitions/list-seasons"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_list_champions(is_id: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List champions of competition through years"
    id: The value of id field returned in .../search or .../competitions/list-default endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/competitions/list-champions"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staffs_get_profile(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get profile of specific staff"
    id: The value of id field returned in .../search or .../staffs/list... endpoints. 
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/staffs/get-profile"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_performance(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players performance of specific club"
    id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-performance"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_performance_summary(is_id: int, seasonid: str=None, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player performance summary or by specific season"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints
        seasonid: The value of id field returned in .../competitions/list-seasons endpoint
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-performance-summary"
    querystring = {'id': is_id, }
    if seasonid:
        querystring['seasonID'] = seasonid
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_market_value(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player's market value"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. 
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-market-value"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs_get_squad(is_id: int, saison_id: int=2022, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get squad of club"
    id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        saison_id: Get squad by season
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/clubs/get-squad"
    querystring = {'id': is_id, }
    if saison_id:
        querystring['saison_id'] = saison_id
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs_get_profile(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get profile of club"
    id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/clubs/get-profile"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs_get_short_info(ids: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short information of one or more clubs"
    ids: The value of id field returned in .../search or .../clubs/list-by-competition endpoints. Separated by coma for multiple values. Ex : 631,383,985,27,etc...
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/clubs/get-short-info"
    querystring = {'ids': ids, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, page: str=None, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for relating players, clubs, competitions, etc... by familiar term or phrase"
    query: Any familiar term or phrase
        page: Page index, for paging purpose
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/search"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_detail_deprecated(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of player
		* Use .../players/get-header-info endpoint instead"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. 
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_achievements(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get achievements of a player"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. 
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-achievements"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_short_info(ids: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short information of one or more players"
    ids: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. Separated by coma for multiple values. Ex : 74842,255755,45660,39381,etc...
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-short-info"
    querystring = {'ids': ids, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_transfer_history(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transfer history of a player"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. 
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-transfer-history"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_profile(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player profile"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. 
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-profile"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_get_performance_detail(competitionid: str, is_id: int, seasonid: int=2021, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player performance detail by specific season and competition"
    competitionid: The value of id field returned in …/search or …/competitions/list-default endpoints.
        id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints
        seasonid: The value of id field returned in .../competitions/list-seasons endpoint
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/players/get-performance-detail"
    querystring = {'competitionID': competitionid, 'id': is_id, }
    if seasonid:
        querystring['seasonID'] = seasonid
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail of specific news"
    id: The value of id field returned in .../news/list... endpoints
        
    """
    url = f"https://transfermarket.p.rapidapi.com/news/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_latest(timestamp: int=None, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news"
    timestamp: Leave empty to load first page, or pass in the value of timestamp field returned right in this endpoint to load the next page.
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/news/list-latest"
    querystring = {}
    if timestamp:
        querystring['timestamp'] = timestamp
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_player(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news relating to specific player"
    id: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/news/list-by-player"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_club(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news relating to specific club"
    id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/news/list-by-club"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_live_table(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live table relating to specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-live-table"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_result(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get result of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-result"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_events(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-events"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_highlights(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get highlights of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-highlights"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_live_ticker(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live ticker of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-live-ticker"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_line_ups(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get line ups of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-line-ups"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_club_comparison(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club comparison of specific match"
    id: The value of id field returned in .../matches/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/get-club-comparison"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list_by_date(date: str=None, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List matches by specific date"
    date: Leave empty to load today matches, or pass in date with format yyyy-MM-dd. Ex : 2020-10-14
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/matches/list-by-date"
    querystring = {}
    if date:
        querystring['date'] = date
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_fifa_world_rankings(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List FIFA world rankings"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-fifa-world-rankings"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_best_fifa_players(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List best FIFA players"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-best-fifa-players"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_golden_boot(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List golden boot"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-golden-boot"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_clean_sheets(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List clean sheets"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-clean-sheets"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_most_valuable_competitions(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List most valuable competitions"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-most-valuable-competitions"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_most_valuable_team(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List most valuable team"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-most-valuable-team"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistic_list_most_valuable_clubs(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List most valuable clubs"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/statistic/list-most-valuable-clubs"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_list_by_club(seasonid: int, is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List transfers relating to specific club"
    seasonid: The value of seasons/id JSON object returned right in this endpoint.
        id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/transfers/list-by-club"
    querystring = {'seasonID': seasonid, 'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_list_market_value(competitionids: str, maxvalue: int=None, minvalue: int=None, playerids: str=None, maxage: int=None, positionids: str=None, clubids: str=None, positiongroup: str=None, offset: int=0, orderbylatestupdate: bool=None, minage: int=None, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List players info, and estimated market value"
    competitionids: The value of id field returned in .../search or .../competitions/list... endpoints. DO NOT use together with clubIds and playerIds parameters. Separated by coma for multiple values. Ex : IT1,GB1,etc...
        maxvalue: The max market value of players. Ex : 151000000
        minvalue: The min market value of players. Ex : 9000000
        playerids: The value of id field returned in .../search or .../clubs/get-squad or .../transfers/list... endpoints. DO NOT use together with clubIds and competitionIds parameters. Separated by coma for multiple values. Ex : 74842,255755,45660,39381,etc...
        maxage: The max age of players to filter (Max 45)
        positionids: One of the following 1-Goalkeeper|2-Sweeper|3-Centre-Back|4-Left-Back|5-Right-Back|6-Defensive Midfield|7-Central Midfield|8-Right Midfield|9-Left Midfield|10-Attacking Midfield|11-Left Winger|12-Right Winger|13-Second Striker|14-Centre-Forward. Separated by coma for multiple values. Ex : 1,6,7,etc...
        clubids: The value of id field returned in .../search or .../clubs/list-by-competition endpoints. DO NOT use together with competitionIds and playerIds parameters. Separated by coma for multiple values. Ex : 631,383,985,27,etc...
        positiongroup: One of the following Abwehr|Mittelfeld|Sturm
        offset: The offset to be ignored, for paging purpose
        orderbylatestupdate: true|false
        minage: The min age of players to filter (Min 14)
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/transfers/list-market-value"
    querystring = {'competitionIds': competitionids, }
    if maxvalue:
        querystring['maxValue'] = maxvalue
    if minvalue:
        querystring['minValue'] = minvalue
    if playerids:
        querystring['playerIds'] = playerids
    if maxage:
        querystring['maxAge'] = maxage
    if positionids:
        querystring['positionIds'] = positionids
    if clubids:
        querystring['clubIds'] = clubids
    if positiongroup:
        querystring['positionGroup'] = positiongroup
    if offset:
        querystring['offset'] = offset
    if orderbylatestupdate:
        querystring['orderByLatestUpdate'] = orderbylatestupdate
    if minage:
        querystring['minAge'] = minage
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_list_records(competitionid: str, positiongroup: str=None, clubid: int=None, offset: int=0, limit: int=30, positionid: int=None, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List records transfers information along with player info, and trade value"
    competitionid: The value of id field returned in .../search or .../competitions/list... endpoints. DO NOT use together with clubID
        positiongroup: One of the following Abwehr|Mittelfeld|Sturm
        clubid: The value of id field returned in .../search or .../clubs/list-by-competition endpoints. DO NOT use together with competitionID
        offset: The offset to be ignored, for paging purpose
        limit: The number of items per response, for paging purpose
        positionid: One of the following 1-Goalkeeper|2-Sweeper|3-Centre-Back|4-Left-Back|5-Right-Back|6-Defensive Midfield|7-Central Midfield|8-Right Midfield|9-Left Midfield|10-Attacking Midfield|11-Left Winger|12-Right Winger|13-Second Striker|14-Centre-Forward
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/transfers/list-records"
    querystring = {'competitionID': competitionid, }
    if positiongroup:
        querystring['positionGroup'] = positiongroup
    if clubid:
        querystring['clubID'] = clubid
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if positionid:
        querystring['positionID'] = positionid
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_list(competitionid: str, positiongroup: str=None, limit: int=30, maxvalue: int=None, positionid: int=None, clubid: int=None, minvalue: int=None, offset: int=0, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List transfers information along with player info, and market value"
    competitionid: The value of id field returned in .../search or .../competitions/list... endpoints. DO NOT use together with clubID
        positiongroup: One of the following Abwehr|Mittelfeld|Sturm
        limit: The number of items per response, for paging purpose
        maxvalue: The max market value of players. Ex : 151000000
        positionid: One of the following 1-Goalkeeper|2-Sweeper|3-Centre-Back|4-Left-Back|5-Right-Back|6-Defensive Midfield|7-Central Midfield|8-Right Midfield|9-Left Midfield|10-Attacking Midfield|11-Left Winger|12-Right Winger|13-Second Striker|14-Centre-Forward
        clubid: The value of id field returned in .../search or .../clubs/list-by-competition endpoints. DO NOT use together with competitionID
        minvalue: The min market value of players. Ex : 9000000
        offset: The offset to be ignored, for paging purpose
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/transfers/list"
    querystring = {'competitionID': competitionid, }
    if positiongroup:
        querystring['positionGroup'] = positiongroup
    if limit:
        querystring['limit'] = limit
    if maxvalue:
        querystring['maxValue'] = maxvalue
    if positionid:
        querystring['positionID'] = positionid
    if clubid:
        querystring['clubID'] = clubid
    if minvalue:
        querystring['minValue'] = minvalue
    if offset:
        querystring['offset'] = offset
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_get_game_plan(is_id: str, seasonid: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game plan by competition and season"
    id: The value of id field returned in .../search or .../competitions/list-default endpoints
        seasonid: The value of id field returned in .../competitions/list-seasons endpoint
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/competitions/get-game-plan"
    querystring = {'id': is_id, 'seasonID': seasonid, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_get_short_info(ids: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short information of one or more competitions"
    ids: The value of id field returned in .../search or .../competitions/list-default endpoints. Separated by coma for multiple values. Ex : L1,L2,L3,GB1,ES1,IT1,etc...
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/competitions/get-short-info"
    querystring = {'ids': ids, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staffs_get_short_info(ids: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short information of one or more staffs"
    ids: The value of id field returned in .../search or .../staffs/list... endpoints. Separated by coma for multiple values. Ex : 60805,39208,17455,38756,72819,48174,etc...
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/staffs/get-short-info"
    querystring = {'ids': ids, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staffs_list_by_competition(is_id: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List staffs relating to specific competition or league"
    id: The value of id field returned in .../search or .../competitions/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/staffs/list-by-competition"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staffs_list_by_club(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List staffs of specific club"
    id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/staffs/list-by-club"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs_get_header_info(is_id: int, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get header info of club"
    id: The value of id field returned in .../search or .../clubs/list-by-competition endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/clubs/get-header-info"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs_list_by_competition(is_id: str, domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List clubs in specific competition"
    id: The value of id field returned in .../search or .../competitions/list... endpoints
        domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/clubs/list-by-competition"
    querystring = {'id': is_id, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_list_default(domain: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List most popular competitions"
    domain: One of the following : com|de|be|es|it|nl|pl|pt|com.tr|world
        
    """
    url = f"https://transfermarket.p.rapidapi.com/competitions/list-default"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

