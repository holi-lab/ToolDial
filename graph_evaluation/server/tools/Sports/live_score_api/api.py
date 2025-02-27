import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def match_events(is_id: str='164008', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting match event like: goals, yellow and red cards and people related to them"
    is_id: The id of the match which events we want to get
        
    """
    url = f"https://live-score-api.p.rapidapi.com/scores/events.json"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting the list of the countries that we have in our database"
    
    """
    url = f"https://live-score-api.p.rapidapi.com/countries/list.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_list(federation_id: int=None, country_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting the list of all supported competitions"
    federation_id: ID of a federation in case we want only competition from certain federation like UEFA or CONMEBOL
        country_id: ID of a country in case we want only competition from certain country like Germany
        
    """
    url = f"https://live-score-api.p.rapidapi.com/competitions/list.json"
    querystring = {}
    if federation_id:
        querystring['federation_id'] = federation_id
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def livescores(competition_id: int=None, country: int=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the API endpoint you get livescores for all current matches"
    competition_id: The ID of the competition for which we want the results
        country: The ID of the country for which we want the live scores
        lang: 2 letter code of a language in which we want the team names (ar, ru...)
        
    """
    url = f"https://live-score-api.p.rapidapi.com/scores/live.json"
    querystring = {}
    if competition_id:
        querystring['competition_id'] = competition_id
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures(page: int=None, competition_id: int=None, date: str=None, lang: str=None, round: str=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting list of scheduled games (calendar)"
    page: The page with results, this endoint uses pagination
        competition_id: The ID of a competition in case we want the fixtures for certain competition only
        date: Filters the fixtures by the date when the matches are scheduled to be played
        lang: 2 letter ISO code of the language in which the response data should be in
        round: The round which fixtures we want it could be a number or code like QF, SF, R16
        team: The ID of a team in case we want all the fixtures for a certain team
        
    """
    url = f"https://live-score-api.p.rapidapi.com/fixtures/matches.json"
    querystring = {}
    if page:
        querystring['page'] = page
    if competition_id:
        querystring['competition_id'] = competition_id
    if date:
        querystring['date'] = date
    if lang:
        querystring['lang'] = lang
    if round:
        querystring['round'] = round
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_statistics(match_id: int=172939, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting match statistics like: possession, offsides, corners and more"
    match_id: The id of the match which events we are getting
        
    """
    url = f"https://live-score-api.p.rapidapi.com/matches/stats.json"
    querystring = {}
    if match_id:
        querystring['match_id'] = match_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history(is_from: str=None, lang: str=None, team: int=None, competition_id: int=None, page: int=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting matches that have finished"
    is_from: Date since when we want to get matches
        lang: 2 letter ISO code of the language in which we want the data to be in
        team: ID of a team in case we want only matches of a certain team
        competition_id: ID of a competition in case we want matches only from a certain competition
        page: Page number with matches as this endpoint uses pagination
        to: Date until when we want to get matches
        
    """
    url = f"https://live-score-api.p.rapidapi.com/scores/history.json"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if lang:
        querystring['lang'] = lang
    if team:
        querystring['team'] = team
    if competition_id:
        querystring['competition_id'] = competition_id
    if page:
        querystring['page'] = page
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def head2head_team_comparison(team2_id: int, team1_id: int, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Comparing 2 teams based on their recent performances"
    team2_id: The id of the second team that we are comparing
        team1_id: The id of the team that we want to compare
        lang: 2 letter iso code in which we want the data in
        
    """
    url = f"https://live-score-api.p.rapidapi.com/teams/head2head.json"
    querystring = {'team2_id': team2_id, 'team1_id': team1_id, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_standings(season: int, competition_id: int, group: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting the classification table of a competition"
    season: The season for which we want it
        competition_id: The id of the competition which standings we want
        group: The id of a specific group in case, the competition has several groups like the Champions League
        lang: 2 letter ISO code of the language in which we want the data in
        
    """
    url = f"https://live-score-api.p.rapidapi.com/leagues/table.json"
    querystring = {'season': season, 'competition_id': competition_id, }
    if group:
        querystring['group'] = group
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def federations_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting the list of the federations AFC, CAF, CONMEBOL, UEFA and more"
    
    """
    url = f"https://live-score-api.p.rapidapi.com/federations/list.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_list(size: int=None, page: int=None, country_id: int=None, federation_id: int=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting a list of the teams from our database"
    size: Between 1 and 100, the number of teams per page
        page: The number of page, as this endpoint uses pagination
        country_id: ID of a country in case we want teams from only one country
        federation_id: ID of a country in case we want teams from only one federation, these will be national teams
        language: 2 letter ISO of the language in which we want the data in
        
    """
    url = f"https://live-score-api.p.rapidapi.com/teams/list.json"
    querystring = {}
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if country_id:
        querystring['country_id'] = country_id
    if federation_id:
        querystring['federation_id'] = federation_id
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-score-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

