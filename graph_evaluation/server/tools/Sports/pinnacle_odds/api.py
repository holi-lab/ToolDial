import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def betting_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a betting status. Checking the Pinnacle server"
    
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/betting-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_special_markets(sport_id: int, is_have_odds: bool=None, league_ids: int=None, event_type: str=None, since: int=None, event_ids: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of special markets. Always first issue a snapshot call and continue with the delta calls. Calls return changes since the provided `since` value. You must always use the since parameter, after the first call. Please note that prematch and live events are different"
    sport_id: Sport id
        is_have_odds: `1` or `0`. You can only get matches for which there are already open odds, or matches that will be given odds in the future
        league_ids: League id
        event_type: Status: `prematch`, `live`  Please note that prematch and live events are different
        since: Since UTC time. Calls return changes since the provided `since` value.
        event_ids: Event id
        
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/special-markets"
    querystring = {'sport_id': sport_id, }
    if is_have_odds:
        querystring['is_have_odds'] = is_have_odds
    if league_ids:
        querystring['league_ids'] = league_ids
    if event_type:
        querystring['event_type'] = event_type
    if since:
        querystring['since'] = since
    if event_ids:
        querystring['event_ids'] = event_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_details(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a event details and history odds. history:[time, value, max bet]. `Period_results - status`: 1 = Event period is settled, 2 = Event period is re-settled, 3 = Event period is cancelled, 4 = Event period is re-settled as cancelled, 5 = Event is deleted"
    event_id: Event id
        
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/details"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_periods(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of periods"
    sport_id: Sport id
        
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/meta-periods"
    querystring = {'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_archive_events(sport_id: int, page_num: int, league_ids: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of archive events. Use pagination"
    sport_id: Sport id
        page_num: Page num
        league_ids: League id
        
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/archive"
    querystring = {'sport_id': sport_id, 'page_num': page_num, }
    if league_ids:
        querystring['league_ids'] = league_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_sports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of sports"
    
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/sports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_markets(sport_id: int, league_ids: int=None, event_type: str=None, event_ids: int=None, is_have_odds: bool=None, since: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of markets. Always first issue a snapshot call and continue with the delta calls. Calls return changes since the provided `since` value.  You must always use the `since` parameter, after starting your program cycle. You can make request without a `since` parameter no more than 15 times in 5 minutes. Please note that `prematch` and `live` events are different"
    sport_id: Sport id
        league_ids: League id
        event_type: Status: `prematch`, `live`  Please note that prematch and live events are different
        event_ids: Event id
        is_have_odds: `1` or `0`. You can only get matches for which there are already open odds, or matches that will be given odds in the future
        since: Since UTC time. Calls return changes since the provided `since` value.
        
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/markets"
    querystring = {'sport_id': sport_id, }
    if league_ids:
        querystring['league_ids'] = league_ids
    if event_type:
        querystring['event_type'] = event_type
    if event_ids:
        querystring['event_ids'] = event_ids
    if is_have_odds:
        querystring['is_have_odds'] = is_have_odds
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_leagues(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of leagues"
    sport_id: Sport id
        
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/kit/v1/leagues"
    querystring = {'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

