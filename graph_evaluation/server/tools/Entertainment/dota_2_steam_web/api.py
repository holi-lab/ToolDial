import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def match_details(match_id: str, matches_requested: str, date_min: str=None, player_name: str=None, hero_id: str=None, skill: str=None, date_max: str=None, account_id: str=None, league_id: str=None, start_at_match_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To retrieve the specific details of a match, use this API:"
    matches_requested: Defaults is 25 matches, this can limit to less
        date_min: date in UTC seconds since Jan 1, 1970 (unix time format)
        player_name: Search matches with a player name, exact match only
        hero_id: Search for matches with a specific hero being played, hero id's are in dota/scripts/npc/npc_heroes.txt in your Dota install directory
        skill: 0 for any, 1 for normal, 2 for high, 3 for very high skill
        date_max: date in UTC seconds since Jan 1, 1970 (unix time format)
        account_id: Steam account id (this is not SteamID, its only the account number portion)
        league_id: matches for a particular league
        start_at_match_id: Start the search at the indicated match id, descending
        
    """
    url = f"https://community-dota-2.p.rapidapi.com/IDOTA2Match_570/GetMatchDetails/V001/"
    querystring = {'match_id': match_id, 'matches_requested': matches_requested, }
    if date_min:
        querystring['date_min'] = date_min
    if player_name:
        querystring['player_name'] = player_name
    if hero_id:
        querystring['hero_id'] = hero_id
    if skill:
        querystring['skill'] = skill
    if date_max:
        querystring['date_max'] = date_max
    if account_id:
        querystring['account_id'] = account_id
    if league_id:
        querystring['league_id'] = league_id
    if start_at_match_id:
        querystring['start_at_match_id'] = start_at_match_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-dota-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_history(format: str='JSON', start_at_match_id: str='27110133', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "will return the latest 25 public matches in JSON format. You can request it in XML format"
    format: XML or JSON
        start_at_match_id: To request the next 25, use the param "start_at_match_id" with one less than the last match number you received
        
    """
    url = f"https://community-dota-2.p.rapidapi.com/IDOTA2Match_570/GetMatchHistory/V001/"
    querystring = {}
    if format:
        querystring['format'] = format
    if start_at_match_id:
        querystring['start_at_match_id'] = start_at_match_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-dota-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

