import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def matches_v2_detail_deprecated(eid: int, category: str, timezone: int=-7, livetable: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of a match
		*The default timezone currently is UTC+7"
    eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live endpoints
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        timezone: One of the following : -11|-10|-9.5|-9|-8.5|-8|-7|-6|-5|-4.5|-4|-3.5|-3|-2.5|-2|-1|0|1|2|3|3.5|4|4.5|5|5.5|5.75|6|6.5|6.75|7|7.5|8|8.5|8.75|9|9.5|9.75|10|10.5|11|11.5|12|12.5|12.75|13|13.75|14
        livetable: true|false - Whether or not include live table relating to current match
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/detail"
    querystring = {'Eid': eid, 'Category': category, }
    if timezone:
        querystring['Timezone'] = timezone
    if livetable:
        querystring['LiveTable'] = livetable
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list_by_sport(category: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news relating to a specific sport by category"
    category: The value of categories -> id JSON object returned in .../news/v2/list endpoint
        page: For paging purpose
        
    """
    url = f"https://livescore6.p.rapidapi.com/news/v2/list-by-sport"
    querystring = {'category': category, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_list_live(category: str, timezone: int=-7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List available live matches at request time
		* Base for Img field is https://lsm-static-prod.livescore.com/medium . Ex : https://lsm-static-prod.livescore.com/medium/enet/9906.png"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        timezone: One of the following : -11|-10|-9.5|-9|-8.5|-8|-7|-6|-5|-4.5|-4|-3.5|-3|-2.5|-2|-1|0|1|2|3|3.5|4|4.5|5|5.5|5.75|6|6.5|6.75|7|7.5|8|8.5|8.75|9|9.5|9.75|10|10.5|11|11.5|12|12.5|12.75|13|13.75|14
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/list-live"
    querystring = {'Category': category, }
    if timezone:
        querystring['Timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_innings(eid: int, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get innings of a match (for "cricket" as category)"
    eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-innings"
    querystring = {'Eid': eid, 'Category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_h2h(category: str, eid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get head to head information in the past"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-h2h"
    querystring = {'Category': category, 'Eid': eid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_table(category: str, eid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get table"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-table"
    querystring = {'Category': category, 'Eid': eid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_scoreboard(eid: int, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get scoreboard of a match"
    eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-scoreboard"
    querystring = {'Eid': eid, 'Category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_info(category: str, eid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get where the match takes place"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-info"
    querystring = {'Category': category, 'Eid': eid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_team_stats(is_id: int, compid: int=65, stype: str='cm', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics relating to a competition"
    id: The value of ID field returned in .../competitions/detail, .../matches/v2/list-by-date, .../matches/v2/list-by-league, .../matches/v2/list-live 
OR Tid field returned in  ../get-table endpoint
        stype: One of the following : cm-Season|ts-Tournament
        
    """
    url = f"https://livescore6.p.rapidapi.com/teams/get-team-stats"
    querystring = {'ID': is_id, }
    if compid:
        querystring['CompId'] = compid
    if stype:
        querystring['Stype'] = stype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_player_stats(is_id: int, type: int=None, stype: str='cm', compid: int=65, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players statistics relating to a competition"
    id: The value of ID field returned in .../competitions/detail, .../matches/v2/list-by-date, .../matches/v2/list-by-league, .../matches/v2/list-live 
OR Tid field returned in  ../get-table endpoint
        type: One of the following : 1-Goals|3-Assists|8-Shots on target|4-Red cards|6-Yellow cards
        stype: One of the following : cm-Season|ts-Tournament
        
    """
    url = f"https://livescore6.p.rapidapi.com/teams/get-player-stats"
    querystring = {'ID': is_id, }
    if type:
        querystring['Type'] = type
    if stype:
        querystring['Stype'] = stype
    if compid:
        querystring['CompId'] = compid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_get_table(is_id: int, type: str='short', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get table of a team"
    id: The value of ID field returned in .../competitions/detail, .../matches/v2/list-by-date, .../matches/v2/list-by-league, .../matches/v2/list-live 
OR Tid field returned in  ../get-table endpoint
        type: One of the following : full|short
        
    """
    url = f"https://livescore6.p.rapidapi.com/teams/get-table"
    querystring = {'ID': is_id, }
    if type:
        querystring['Type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team information, recent and up coming matches"
    id: The value of ID field returned in .../competitions/detail, .../matches/v2/list-by-date, .../matches/v2/list-by-league, .../matches/v2/list-live 
OR Tid field returned in  ../get-table endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/teams/detail"
    querystring = {'ID': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_get_team_stats(compid: int, type: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get teams statistics in a competition"
    compid: The value of Stages -> CompId field returned in .../leagues/v2/list or .../leagues/v2/get-table or .../matches/v2/list-by-date or .../matches/v2/list-by-league or .../matches/v2/list-live endpoint
        type: One of the following : 10-goals scored|7-Goals conceded|1-Average possession|21-Shots on target|22-Total shots|16-Red cards|23-Yellow cards
        
    """
    url = f"https://livescore6.p.rapidapi.com/competitions/get-team-stats"
    querystring = {'CompId': compid, }
    if type:
        querystring['Type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_get_player_stats(compid: int, type: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players statistics in a competition"
    compid: The value of Stages -> CompId field returned in .../leagues/v2/list or .../leagues/v2/get-table or .../matches/v2/list-by-date or .../matches/v2/list-by-league or .../matches/v2/list-live endpoint
        type: One of the following : 1-Goals|3-Assists|8-Shots on target|4-Red cards|6-Yellow cards
        
    """
    url = f"https://livescore6.p.rapidapi.com/competitions/get-player-stats"
    querystring = {'CompId': compid, }
    if type:
        querystring['Type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_get_table(compid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get table relating to a competition"
    compid: The value of Stages -> CompId field returned in .../leagues/v2/list or .../leagues/v2/get-table or .../matches/v2/list-by-date or .../matches/v2/list-by-league or .../matches/v2/list-live endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/competitions/get-table"
    querystring = {'CompId': compid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_detail(compid: int, timezone: int=-7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition information, recent and up coming matches"
    compid: The value of Stages -> CompId field returned in .../leagues/v2/list or .../leagues/v2/get-table or .../matches/v2/list-by-date or .../matches/v2/list-by-league or .../matches/v2/list-live endpoint
        timezone: One of the following : -11|-10|-9.5|-9|-8.5|-8|-7|-6|-5|-4.5|-4|-3.5|-3|-2.5|-2|-1|0|1|2|3|3.5|4|4.5|5|5.5|5.75|6|6.5|6.75|7|7.5|8|8.5|8.75|9|9.5|9.75|10|10.5|11|11.5|12|12.5|12.75|13|13.75|14
        
    """
    url = f"https://livescore6.p.rapidapi.com/competitions/detail"
    querystring = {'CompId': compid, }
    if timezone:
        querystring['Timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list_by_league_deprecated(category: str, league: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all matches by specific league"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        league: The value of Ccd field at top level that responded in leagues/list API
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/list-by-league"
    querystring = {'category': category, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_get_table_deprecated(p: str, category: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get table of related match"
    p: The value of Stages/Events/IDs/p field responded in .../matches/list-by-league or .../matches/list-by-date or .../matches/list-live endpoint
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        id: The value of Stages/Events/IDs/ID field responded in .../matches/list-by-league or .../matches/list-by-date or .../matches/list-live endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/get-table"
    querystring = {'p': p, 'category': category, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_list_by_date(category: str, date: str, timezone: int=-7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all matches by specific date
		* Base for Img field is https://lsm-static-prod.livescore.com/medium . Ex : https://lsm-static-prod.livescore.com/medium/enet/9906.png"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        date: The date string with following format yyyyMMdd
        timezone: One of the following : -11|-10|-9.5|-9|-8.5|-8|-7|-6|-5|-4.5|-4|-3.5|-3|-2.5|-2|-1|0|1|2|3|3.5|4|4.5|5|5.5|5.75|6|6.5|6.75|7|7.5|8|8.5|8.75|9|9.5|9.75|10|10.5|11|11.5|12|12.5|12.75|13|13.75|14
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/list-by-date"
    querystring = {'Category': category, 'Date': date, }
    if timezone:
        querystring['Timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_detail_deprecated(is_id: int, p: int, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of a match"
    id: The value of Stages/Events/IDs/ID field responded in .../matches/list-by-league or .../matches/list-by-date or .../matches/list-live endpoint
        p: The value of Stages/Events/IDs/p field responded in .../matches/list-by-league or .../matches/list-by-date or .../matches/list-live endpoint
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/detail"
    querystring = {'id': is_id, 'p': p, 'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_lineups(eid: int, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match line-ups"
    eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-lineups"
    querystring = {'Eid': eid, 'Category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_deprecated(category: str, key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all news of specific category"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        key: Pass empty for init request, or value of key field returned right in this API to load the next records
        
    """
    url = f"https://livescore6.p.rapidapi.com/news/list"
    querystring = {'category': category, }
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news relating to all sports at once"
    
    """
    url = f"https://livescore6.p.rapidapi.com/news/v2/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news content"
    id: The value of articles -> id field returned in .../news/v2/list or .../news/v2/list-by-sport
        
    """
    url = f"https://livescore6.p.rapidapi.com/news/v2/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_list_by_league(category: str, ccd: str, scd: str='group-b', timezone: int=-7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all matches by specific league and table
		* Base for Img field is https://lsm-static-prod.livescore.com/medium . Ex : https://lsm-static-prod.livescore.com/medium/enet/9906.png"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        ccd: The value of Ccd field that responded in .../leagues/v2/list or .../matches/v2/list-by-date endpoints.
        scd: The value of Scd field that responded in .../leagues/v2/list or .../matches/v2/list-by-date endpoints.
        timezone: One of the following : -11|-10|-9.5|-9|-8.5|-8|-7|-6|-5|-4.5|-4|-3.5|-3|-2.5|-2|-1|0|1|2|3|3.5|4|4.5|5|5.5|5.75|6|6.5|6.75|7|7.5|8|8.5|8.75|9|9.5|9.75|10|10.5|11|11.5|12|12.5|12.75|13|13.75|14
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/list-by-league"
    querystring = {'Category': category, 'Ccd': ccd, }
    if scd:
        querystring['Scd'] = scd
    if timezone:
        querystring['Timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list_live_deprecated(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List available live matches at request time"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/list-live"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list_by_date_deprecated(date: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all matches by specific date"
    date: The date string with following format yyyy-MM-dd
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/list-by-date"
    querystring = {'date': date, 'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_pregame_form(category: str, eid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pregame form of competitors"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-pregame-form"
    querystring = {'Category': category, 'Eid': eid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_statistics(category: str, eid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics of a match"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-statistics"
    querystring = {'Category': category, 'Eid': eid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_commentary(eid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments in a match"
    eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-commentary"
    querystring = {'Eid': eid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_list_deprecated(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all leagues by category (football, cricket, basketball, tennis, hockey)"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/leagues/list"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_v2_list(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all leagues by category (football, cricket, basketball, tennis, hockey)
		* Base for Img field is https://lsm-static-prod.livescore.com/medium . Ex : https://lsm-static-prod.livescore.com/medium/enet/9906.png"
    category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/leagues/v2/list"
    querystring = {'Category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues_v2_get_table(ccd: str, category: str, scd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get table league by category (football, cricket, basketball, tennis, hockey) and competition
		* Base for Img field is https://lsm-static-prod.livescore.com/medium . Ex : https://lsm-static-prod.livescore.com/medium/enet/9906.png"
    ccd: The value of Ccd field returned in .../leagues/v2/list endpoint
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        scd: The value of Scd field returned in .../leagues/v2/list endpoint
        
    """
    url = f"https://livescore6.p.rapidapi.com/leagues/v2/get-table"
    querystring = {'Ccd': ccd, 'Category': category, 'Scd': scd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_v2_get_incidents(eid: int, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get incidents in a match"
    eid: The value of Eid field returned in .../matches/v2/list-by-league or .../matches/v2/list-by-date or .../matches/v2/list-live or .../competitions/detail endpoint
        category: One of the followings : soccer|cricket|basketball|tennis|hockey
        
    """
    url = f"https://livescore6.p.rapidapi.com/matches/v2/get-incidents"
    querystring = {'Eid': eid, 'Category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

