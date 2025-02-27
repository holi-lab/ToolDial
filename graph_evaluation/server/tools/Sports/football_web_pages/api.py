import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def appearances(team: int=1, comp: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Appearances"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/appearances.json"
    querystring = {}
    if team:
        querystring['team'] = team
    if comp:
        querystring['comp'] = comp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions(include: str='rounds', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Competitions"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/competitions.json"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def form_guide(comp: int=1, type: str=None, count: str=None, team: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Form Guide"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/form-guide.json"
    querystring = {}
    if comp:
        querystring['comp'] = comp
    if type:
        querystring['type'] = type
    if count:
        querystring['count'] = count
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goalscorers(comp: int=1, team: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Goalscorers"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/goalscorers.json"
    querystring = {}
    if comp:
        querystring['comp'] = comp
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_table(comp: int=1, team: int=1, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "League Table"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/league-table.json"
    querystring = {}
    if comp:
        querystring['comp'] = comp
    if team:
        querystring['team'] = team
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def records(team: int=1, comp: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Records"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/records.json"
    querystring = {}
    if team:
        querystring['team'] = team
    if comp:
        querystring['comp'] = comp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rounds(comp: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rounds"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/rounds.json"
    querystring = {}
    if comp:
        querystring['comp'] = comp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sequences(team: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sequences"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/sequences.json"
    querystring = {'team': team, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(comp: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Teams"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/teams.json"
    querystring = {}
    if comp:
        querystring['comp'] = comp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match(match: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Match"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/match.json"
    querystring = {'match': match, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vidiprinter(comp: int=1, team: int=1, date: str='2020-09-26', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vidiprinter"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/vidiprinter.json"
    querystring = {}
    if comp:
        querystring['comp'] = comp
    if team:
        querystring['team'] = team
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attendances(team: int=1, sort: str=None, type: str=None, comp: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Attendances"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/attendances.json"
    querystring = {}
    if team:
        querystring['team'] = team
    if sort:
        querystring['sort'] = sort
    if type:
        querystring['type'] = type
    if comp:
        querystring['comp'] = comp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_progress(team: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "League Progress"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/league-progress.json"
    querystring = {'team': team, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_results(team: int=1, comp: int=1, round: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fixtures/Results"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/fixtures-results.json"
    querystring = {}
    if team:
        querystring['team'] = team
    if comp:
        querystring['comp'] = comp
    if round:
        querystring['round'] = round
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team(team: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team"
    
    """
    url = f"https://football-web-pages1.p.rapidapi.com/team.json"
    querystring = {'team': team, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-web-pages1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

