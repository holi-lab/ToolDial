import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def starting_grid_rankings(race: int, team: int=None, driver: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the starting grid for a race.
		
		> All the parameters of this endpoint can be used together."
    race: The id of the race
        team: The id of the team
        driver: The id of the driver
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/rankings/startinggrid"
    querystring = {'race': race, }
    if team:
        querystring['team'] = team
    if driver:
        querystring['driver'] = driver
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def races(circuit: int=None, date: str=None, last: int=None, type: str='race', season: int=2021, is_id: int=None, timezone: str=None, competition: int=23, next: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available **races** for a competition.
		
		For all requests to races you can add the query parameter `timezone` to your request in order to retrieve the list of races in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint `timezone`
		
		**Available Status**
		* Live
		* Completed
		* Cancelled
		* Postponed
		* Scheduled
		
		**Available Types**
		* Race
		* Sprint
		* 1st Practice
		* 2nd Practice
		* 3rd Practice
		* 1st Qualifying
		* 2nd Qualifying
		* 3rd Qualifying
		
		> This endpoint requires at least one of these parameters `id`, `date`, `next`, `last` and `season`.
		
		> All the parameters of this endpoint can be used together."
    date: A valid date
        last: The X last races
        type: The type of the race
        season: The season of the race
        is_id: The id of the race
        timezone: A valid timezone
        competition: The id of the competition
        next: The X next races
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/races"
    querystring = {}
    if circuit:
        querystring['circuit'] = circuit
    if date:
        querystring['date'] = date
    if last:
        querystring['last'] = last
    if type:
        querystring['type'] = type
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if timezone:
        querystring['timezone'] = timezone
    if competition:
        querystring['competition'] = competition
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pit_stops(race: int, team: int=None, driver: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of pit stops made by all drivers during a race.
		
		> All the parameters of this endpoint can be used together."
    race: The id of the race
        team: The id of the team
        driver: The id of the driver
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/pitstops"
    querystring = {'race': race, }
    if team:
        querystring['team'] = team
    if driver:
        querystring['driver'] = driver
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fastest_laps_rankings(race: int, driver: str=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the ranking of the fastest laps for a race.
		
		> All the parameters of this endpoint can be used together."
    race: The id of the race
        driver: The id of the driver
        team: The id of the team
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/rankings/fastestlaps"
    querystring = {'race': race, }
    if driver:
        querystring['driver'] = driver
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_competitions(search: str='austra', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allow to search for a competition name"
    search: Allow to search for a competition name
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/competitions"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(is_id: int=1, search: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available competitions.
		
		The team `id` are **unique** in the API and teams keep it across all `seasons`
		
		**Sample `logo` of a team :**
		
		![team](https://media.api-sports.io/formula-1/teams/5.png)
		
		> All the parameters of this endpoint can be used together."
    is_id: The id of the team
        search: Allow to search for a team name
        name: The name of the team
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/teams"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_rankings(season: int, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the teams rankings for a season.
		
		> All the parameters of this endpoint can be used together."
    season: The season
        team: The id of the team
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/rankings/teams"
    querystring = {'season': season, }
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def races_rankings(race: int, team: int=None, driver: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the rankings for a race.
		
		> All the parameters of this endpoint can be used together."
    race: The id of the race
        team: The id of the team
        driver: The id of the driver
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/rankings/races"
    querystring = {'race': race, }
    if team:
        querystring['team'] = team
    if driver:
        querystring['driver'] = driver
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drivers(search: str='lewi', name: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available drivers.
		
		The driver `id` are **unique** in the API and drivers keep it across all `seasons`
		
		**Sample `image` of a driver :**
		
		![circuit](https://media.api-sports.io/formula-1/drivers/20.png)
		
		> All the parameters of this endpoint can be used together.
		
		> This endpoint require at least one parameter."
    search: Allow to search for a driver name
        name: The name of the driver
        is_id: The id of the driver
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/drivers"
    querystring = {}
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions(search: str=None, name: str=None, city: str=None, country: str=None, is_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available competitions.
		
		The competition `id` are **unique** in the API and competitions keep it across all `seasons`
		
		> All the parameters of this endpoint can be used together."
    search: Allow to search for a competition name
        name: The name of the competition
        city: The name of the city
        country: The name of the country
        is_id: The id of the competition
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/competitions"
    querystring = {}
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if city:
        querystring['city'] = city
    if country:
        querystring['country'] = country
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def circuits(competition: int=None, search: str=None, is_id: int=1, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available circuits.
		
		The circuit `id` are **unique** in the API and circuits keep it across all `seasons`
		
		**Sample `image` of a circuit :**
		
		![circuit](https://media.api-sports.io/formula-1/circuits/1.png)
		
		> All the parameters of this endpoint can be used together."
    competition: The id of the competition
        search: Allow to search for a circuit name
        is_id: The id of the circuit
        name: The name of the circuit
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/circuits"
    querystring = {}
    if competition:
        querystring['competition'] = competition
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all seasons available.
		
		All seasons are only **4-digit keys**. All results can be used in other endpoints as filters.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-formula-1.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available timezone to be used in the races endpoint.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-formula-1.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drivers_rankings(season: int, team: str=None, driver: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the drivers rankings for a season.
		
		> All the parameters of this endpoint can be used together."
    season: The season
        team: The id of the team
        driver: The id of the driver
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/rankings/drivers"
    querystring = {'season': season, }
    if team:
        querystring['team'] = team
    if driver:
        querystring['driver'] = driver
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_circuits(search: str='silver', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allow to search for a circuit name"
    search: Allow to search for a circuit name
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/circuits"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_drivers(search: str='lewi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allow to search for a driver name"
    search: Allow to search for a driver name
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/drivers"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_teams(search: str='petron', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allow to search for a team name"
    search: Allow to search for a team name
        
    """
    url = f"https://api-formula-1.p.rapidapi.com/teams"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-formula-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

