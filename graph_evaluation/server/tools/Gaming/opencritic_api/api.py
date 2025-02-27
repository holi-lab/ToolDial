import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reviews_for_game_legacy_fix(is_id: int, skip: int=20, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets reviews for a specific Game ID. Uses /review/game/{id} for legacy support."
    sort: Sorts reviews based on one of the following:

- newest (newest reviews first, by publishedDate of the review)
- oldest (oldest reviews first, by publishedDate of the review)
- score-high (highest scoring reviews first, with unscored reviews last)
- score-low (lowest scoring reviews first, with unscored reviews last)
- popularity (based on publication popularity)
- blend (reviews within the last 3.5 days first; otherwise, sort by popularity)
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/review/game/{is_id}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game data"
    id: The OpenCritic Game ID
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_hall_of_fame(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top 12 games on the OpenCritic Hall of Fame for the current year (usually changes in early February)."
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/hall-of-fame"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def general_search(criteria: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search a combination of Outlet, Game, and Author by String"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/meta/search"
    querystring = {'criteria': criteria, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_outlet_profiles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Outlet profiles"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/outlet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_search(criteria: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches OpenCritic for games with similar names/titles. Search results are based on inverse trigram distance, with 0 being a perfect match and 1 being no shared/overlapping trigrams. See the [pg_trgm](https://www.postgresql.org/docs/current/pgtrgm.html) library for more information."
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/search"
    querystring = {'criteria': criteria, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_popular(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the most popular games on OpenCritic, using a blend of page views, recent reviews, and the notoriety of publications that have reviewed it"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/popular"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game(platforms: str='ps4,ps5', sort: str=None, order: str=None, skip: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for searching game listings"
    platforms: Platforms to search in a comma-separated list. Valid values are:

all,ps4,xb1,pc,wii-u,vita,switch,oculus,vive,psvr,3ds,xbsx,ps5,stadia,vive
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/game"
    querystring = {}
    if platforms:
        querystring['platforms'] = platforms
    if sort:
        querystring['sort'] = sort
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def score_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets OpenCritic Score Formats"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/score-format"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def author_search(criteria: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for an Author"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/author/search"
    querystring = {'criteria': criteria, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_reviewed_this_week(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the most reviewed games this week"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/reviewed-this-week"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genre(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets OpenCritic genres"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/genre"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platform(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets OpenCritic Platforms"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/platform"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_for_outlet(is_id: int, skip: int=20, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets reviews for a specific Outlet ID"
    sort: Sorts reviews based on one of the following:

- newest (newest reviews first, by publishedDate of the review)
- oldest (oldest reviews first, by publishedDate of the review)
- score-high (highest scoring reviews first, with unscored reviews last)
- score-low (lowest scoring reviews first, with unscored reviews last)
- popularity (based on publication popularity)
- blend (reviews within the last 3.5 days first; otherwise, sort by popularity)
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/review/outlet/{is_id}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_for_author(is_id: int, skip: int=20, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets reviews for a specific Author ID"
    sort: Sorts reviews based on one of the following:

- newest (newest reviews first, by publishedDate of the review)
- oldest (oldest reviews first, by publishedDate of the review)
- score-high (highest scoring reviews first, with unscored reviews last)
- score-low (lowest scoring reviews first, with unscored reviews last)
- popularity (based on publication popularity)
- blend (reviews within the last 3.5 days first; otherwise, sort by popularity)
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/review/author/{is_id}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def outlet_profile(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a given outlet's profile"
    id: The Outlet's ID
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/outlet/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def author_profile(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets an Author's profile data on OpenCritic"
    id: Author ID
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/author/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_reviewed_today(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the 10 games that have been most reviewed today."
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/reviewed-today"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_hall_of_fame_year(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the Hall of Fame for a specific year"
    year: Year between 2016 and the current calendar year
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/hall-of-fame/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_recently_released(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the 8 most recently released major titles on OpenCritic"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/recently-released"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_upcoming(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the next 8 upcoming major titles on OpenCritic"
    
    """
    url = f"https://opencritic-api.p.rapidapi.com/game/upcoming"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_for_game(is_id: int, sort: str=None, skip: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets reviews for a specific Game ID"
    sort: Sorts reviews based on one of the following:

- newest (newest reviews first, by publishedDate of the review)
- oldest (oldest reviews first, by publishedDate of the review)
- score-high (highest scoring reviews first, with unscored reviews last)
- score-low (lowest scoring reviews first, with unscored reviews last)
- popularity (based on publication popularity)
- blend (reviews within the last 3.5 days first; otherwise, sort by popularity)
        
    """
    url = f"https://opencritic-api.p.rapidapi.com/reviews/game/{is_id}"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencritic-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

