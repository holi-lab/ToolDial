import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tags_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Tag.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/tags/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Store.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/stores/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creators_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Person.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/creators/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platforms_list(page: int=None, ordering: str=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page: A page number within the paginated result set.
        ordering: Which field to use when ordering the results.
        page_size: Number of results to return per page.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/platforms"
    querystring = {}
    if page:
        querystring['page'] = page
    if ordering:
        querystring['ordering'] = ordering
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Game.
        id: An ID or a slug identifying this Game.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def developers_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/developers"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platforms_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Platform.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/platforms/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_stores_list(game_pk: str, page_size: int=None, ordering: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        ordering: Which field to use when ordering the results.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{game_pk}/stores"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if ordering:
        querystring['ordering'] = ordering
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creators_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/creators"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genres_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Genre.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/genres/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_youtube_read(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: An ID or a slug identifying this Game.
        id: A unique integer value identifying this Game.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{is_id}/youtube"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platforms_lists_parents_list(ordering: str=None, page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For instance, for PS2 and PS4 the “parent platform” is PlayStation."
    ordering: Which field to use when ordering the results.
        page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/platforms/lists/parents"
    querystring = {}
    if ordering:
        querystring['ordering'] = ordering
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/tags"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publishers_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Publisher.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/publishers/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_additions_list(game_pk: str, page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{game_pk}/additions"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores_list(page: int=None, ordering: str=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page: A page number within the paginated result set.
        ordering: Which field to use when ordering the results.
        page_size: Number of results to return per page.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/stores"
    querystring = {}
    if page:
        querystring['page'] = page
    if ordering:
        querystring['ordering'] = ordering
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_suggested_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Game.
        id: An ID or a slug identifying this Game.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{is_id}/suggested"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_twitch_read(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: An ID or a slug identifying this Game.
        id: A unique integer value identifying this Game.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{is_id}/twitch"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genres_list(ordering: str=None, page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    ordering: Which field to use when ordering the results.
        page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/genres"
    querystring = {}
    if ordering:
        querystring['ordering'] = ordering
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_reddit_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Game.
        id: An ID or a slug identifying this Game.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{is_id}/reddit"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_achievements_read(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: An ID or a slug identifying this Game.
        id: A unique integer value identifying this Game.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{is_id}/achievements"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_screenshots_list(game_pk: str, page_size: int=None, ordering: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        ordering: Which field to use when ordering the results.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{game_pk}/screenshots"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if ordering:
        querystring['ordering'] = ordering
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_movies_read(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: An ID or a slug identifying this Game.
        id: A unique integer value identifying this Game.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{is_id}/movies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_development_team_list(game_pk: str, ordering: str=None, page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    ordering: Which field to use when ordering the results.
        page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{game_pk}/development-team"
    querystring = {}
    if ordering:
        querystring['ordering'] = ordering
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publishers_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/publishers"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_game_series_list(game_pk: str, page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{game_pk}/game-series"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creator_roles_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/creator-roles"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_list(dates: str=None, creators: str=None, parent_platforms: str=None, search: str=None, exclude_collection: int=None, exclude_game_series: bool=None, metacritic: str=None, platforms: str=None, stores: str=None, page: int=None, search_precise: bool=None, genres: str=None, page_size: int=None, developers: str=None, search_exact: bool=None, updated: str=None, publishers: str=None, platforms_count: int=None, exclude_additions: bool=None, exclude_parents: bool=None, tags: str=None, ordering: str=None, exclude_stores: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    dates: Filter by a release date, for example: `2010-01-01,2018-12-31.1960-01-01,1969-12-31`.
        creators: Filter by creators, for example: `78,28` or `cris-velasco,mike-morasky`.
        parent_platforms: Filter by parent platforms, for example: `1,2,3`.
        search: Search query.
        exclude_collection: Exclude games from a particular collection, for example: `123`.
        exclude_game_series: Exclude games which included in a game series.
        metacritic: Filter by a metacritic rating, for example: `80,100`.
        platforms: Filter by platforms, for example: `4,5`.
        stores: Filter by stores, for example: `5,6`.
        page: A page number within the paginated result set.
        search_precise: Disable fuzziness for the search query.
        genres: Filter by genres, for example: `4,51` or `action,indie`.
        page_size: Number of results to return per page.
        developers: Filter by developers, for example: `1612,18893` or `valve-software,feral-interactive`.
        search_exact: Mark the search query as exact.
        updated: Filter by an update date, for example: `2020-12-01,2020-12-31`.
        publishers: Filter by publishers, for example: `354,20987` or `electronic-arts,microsoft-studios`.
        platforms_count: Filter by platforms count, for example: `1`.
        exclude_additions: Exclude additions.
        exclude_parents: Exclude games which have additions.
        tags: Filter by tags, for example: `31,7` or `singleplayer,multiplayer`.
        ordering: Available fields: `name`, `released`, `added`, `created`, `updated`, `rating`, `metacritic`. You can reverse the sort order adding a hyphen, for example: `-released`.
        exclude_stores: Exclude stores, for example: `5,6`.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games"
    querystring = {}
    if dates:
        querystring['dates'] = dates
    if creators:
        querystring['creators'] = creators
    if parent_platforms:
        querystring['parent_platforms'] = parent_platforms
    if search:
        querystring['search'] = search
    if exclude_collection:
        querystring['exclude_collection'] = exclude_collection
    if exclude_game_series:
        querystring['exclude_game_series'] = exclude_game_series
    if metacritic:
        querystring['metacritic'] = metacritic
    if platforms:
        querystring['platforms'] = platforms
    if stores:
        querystring['stores'] = stores
    if page:
        querystring['page'] = page
    if search_precise:
        querystring['search_precise'] = search_precise
    if genres:
        querystring['genres'] = genres
    if page_size:
        querystring['page_size'] = page_size
    if developers:
        querystring['developers'] = developers
    if search_exact:
        querystring['search_exact'] = search_exact
    if updated:
        querystring['updated'] = updated
    if publishers:
        querystring['publishers'] = publishers
    if platforms_count:
        querystring['platforms_count'] = platforms_count
    if exclude_additions:
        querystring['exclude_additions'] = exclude_additions
    if exclude_parents:
        querystring['exclude_parents'] = exclude_parents
    if tags:
        querystring['tags'] = tags
    if ordering:
        querystring['ordering'] = ordering
    if exclude_stores:
        querystring['exclude_stores'] = exclude_stores
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def developers_read(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: A unique integer value identifying this Developer.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/developers/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_parent_games_list(game_pk: str, page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://newgamerapi.p.rapidapi.com/games/{game_pk}/parent-games"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newgamerapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

