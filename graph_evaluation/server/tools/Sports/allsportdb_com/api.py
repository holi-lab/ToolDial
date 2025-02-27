import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_competitions(is_id: int=None, continent: str=None, continentid: int=None, sport: str=None, sportid: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of competitions (at least one parameter required: id, name, sportId, sport)"
    is_id: competition object Id
        continent: continent name
        continentid: continent object Id, refer to Get Continents method
        sport: sport name
        sportid: sport object Id, refer to Get Sports method
        name: competition name
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/competitions"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if continent:
        querystring['continent'] = continent
    if continentid:
        querystring['continentId'] = continentid
    if sport:
        querystring['sport'] = sport
    if sportid:
        querystring['sportId'] = sportid
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_continents(name: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of continents"
    name: continent name
        is_id: continent object Id
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/continents"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries(name: str=None, is_id: int=None, continent: str=None, continentid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of countries"
    name: country name
        is_id: country object Id
        continent: continent name
        continentid: continent object Id, refer to Get Continents method
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/countries"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if continent:
        querystring['continent'] = continent
    if continentid:
        querystring['continentId'] = continentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sports(season: str=None, is_id: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of sports"
    season: season ('sumer', 'winter' or 'other')
        is_id: sport object Id
        name: sport name
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/sports"
    querystring = {}
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_games_types(is_id: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of games types (Olympic Games, Universiade, etc.)"
    is_id: games type object Id
        name: games type name
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/gamestypes"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_region_flag(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns region flag image"
    id: region object Id
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/regionflag"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_locations(regionid: int=None, countryid: int=None, country: str=None, region: str=None, name: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of locations (at least one parameter required: id, name, countryId, country)"
    regionid: region object Id, refer to Get Regions method
        countryid: country object Id, refer to Get Countries method
        country: country name
        region: region name
        name: location name
        is_id: location object Id
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/locations"
    querystring = {}
    if regionid:
        querystring['regionId'] = regionid
    if countryid:
        querystring['countryId'] = countryid
    if country:
        querystring['country'] = country
    if region:
        querystring['region'] = region
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_calendar(countryid: int=None, continent: str=None, competitionid: int=None, objecttype: int=0, gamestypeid: int=None, continentid: int=None, sport: str=None, dateto: str=None, region: str=None, datefrom: str=None, regionid: int=None, gamestype: str=None, week: int=None, page: int=None, sportid: int=None, competition: str=None, location: str=None, country: str=None, locationid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of events and games (required parameters: week or dateFrom and dateTo, results limited to 50 records)"
    countryid: country object Id, refer to Get Countries method
        continent: continent name
        competitionid: competition object Id, refer to Get Competitions method
        objecttype: object type (1 = event, 2 = games, 0 = both)
        gamestypeid: games type object Id, refer to Get Games Types method
        continentid: continent object Id, refer to Get Continents method
        sport: sport name
        dateto: period end date
        region: region name
        datefrom: period start date
        regionid: region object Id, refer to Get Regions method
        gamestype: games type name
        week: week index (for example, 0 = this week, 1 = next week, -1 = last week, etc.)
        page: page number in case result limit is reached
        sportid: sport object Id, refer to Get Sports method
        competition: competition name
        location: location name
        country: country name
        locationid: location object Id, refer to Get Locations method
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/calendar"
    querystring = {}
    if countryid:
        querystring['countryId'] = countryid
    if continent:
        querystring['continent'] = continent
    if competitionid:
        querystring['competitionId'] = competitionid
    if objecttype:
        querystring['objectType'] = objecttype
    if gamestypeid:
        querystring['gamesTypeId'] = gamestypeid
    if continentid:
        querystring['continentId'] = continentid
    if sport:
        querystring['sport'] = sport
    if dateto:
        querystring['dateTo'] = dateto
    if region:
        querystring['region'] = region
    if datefrom:
        querystring['dateFrom'] = datefrom
    if regionid:
        querystring['regionId'] = regionid
    if gamestype:
        querystring['gamesType'] = gamestype
    if week:
        querystring['week'] = week
    if page:
        querystring['page'] = page
    if sportid:
        querystring['sportId'] = sportid
    if competition:
        querystring['competition'] = competition
    if location:
        querystring['location'] = location
    if country:
        querystring['country'] = country
    if locationid:
        querystring['locationId'] = locationid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_country_flag(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns counry flag image"
    id: country object Id
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/countryflag"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_regions(name: str=None, is_id: int=None, country: str=None, countryid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of country regions, United Kingdom only (at least one parameter required: id, name, countryId, country)"
    name: region name
        is_id: region object Id
        country: country name
        countryid: country object Id, refer to Get Countries method
        
    """
    url = f"https://allsportdb-com.p.rapidapi.com/regions"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    if countryid:
        querystring['countryId'] = countryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "allsportdb-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

