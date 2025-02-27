import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_a_list_of_supported_countries_regions_for_best_podcasts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns a dictionary of country codes (e.g., us, gb...) & country names (United States, United Kingdom...). The country code is used in the query parameter "region" of the /api/v1/best_podcasts endpoint."
    
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_recommendations_for_a_podcast(is_id: str, safe_mode: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Up to 8 podcast recommendations based on the given podcast id"
    id: Podcast id
        safe_mode: Whether or not to exclude podcasts/episodes with explicit language. 1 is yes and 0 is no. Default: 0.
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/podcasts/{is_id}/recommendations"
    querystring = {}
    if safe_mode:
        querystring['safe_mode'] = safe_mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_recommendations_for_an_episode(is_id: str, safe_mode: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Up to 8 episode recommendations based on the given episode id"
    id: Episode id
        safe_mode: Whether or not to exclude podcasts/episodes with explicit language. 1 is yes and 0 is no. Default: 0.
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/episodes/{is_id}/recommendations"
    querystring = {}
    if safe_mode:
        querystring['safe_mode'] = safe_mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_random_podcast_episode(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Good luck!"
    
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/just_listen"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_curated_list_of_podcasts_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get the id of a curated list from /api/v1/search?type=curated or /api/v1/curated_podcasts"
    id: id for a specific curated list of podcasts. You can get the id from the response of /api/v1/search?type=curated or /api/v1/curated_podcasts
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/curated_podcasts/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_curated_lists_of_podcasts(page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A bunch of curated lists from online media. For each list, you'll get basic info of up to 5 podcasts. To get detailed meta data of all podcasts in a specific list, you need to use /api/v1/curated_podcasts/{id}. We constantly add new curated lists to the database."
    page: Page number of curated lists. Default: 1
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/curated_podcasts"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_meta_data_for_a_podcast_by_id(is_id: str, sort: str='recent_first', next_episode_pub_date: int=1479154463000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch meta data for a specific podcast. You can get id from search api's response."
    id: id for a specific podcast. You can get id from search api's response.
        sort: How do you want to sort the episodes of this podcast? Available options: recent_first (default), oldest_first
        next_episode_pub_date: Used for episodes pagination. Typically, it's the value of next_episode_pub_date from the response of last request. If not specified, just return latest 10 episodes or oldest 10 episodes, depending on the value of "sort" parameter.
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/podcasts/{is_id}"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if next_episode_pub_date:
        querystring['next_episode_pub_date'] = next_episode_pub_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def full_text_search(q: str, type: str='episode', genre_ids: str='68,82', language: str='English', safe_mode: int=1, sort_by_date: int=0, offset: int=0, ocid: str=None, ncid: str=None, only_in: str='title', len_max: int=10, len_min: int=2, published_after: int=1390190241000, published_before: int=1490190241000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search episodes or podcasts"
    q: Search term
        type: What to search: "episode" (default), "podcast", or "curated"? Note: "curated" is for curated lists of podcasts.
        genre_ids: A comma-delimited string of a list of genre ids. You can find the id and the name of all genres in the response of /api/v1/genres . It works only when "type" is episode or podcast.
        language: Limit search results to a specific language. If not specified, it'll be any language. You can get a list of supported languages from /api/v1/languages endpoint. It works only when "type" is episode or podcast.
        safe_mode: Whether or not to exclude podcasts/episodes with explicit language. 1 is yes and 0 is no. Default: 0. It works only when "type" is episode or podcast.
        sort_by_date: Sort by date or not? If 1, sort by date. If 0 (default), sort by relevance.
        offset: Offset for search results, for pagination. You'll use next_offset from response for this parameter.
        ocid: A podcast id (the podcast_id field in response). This parameter is to limit search results in a specific podcast. It works only when "type" is episode.
        ncid: A podcast id (the podcast_id field in response). This parameter is to exclude search results from a specific podcast. It works only when "type" is episode.
        only_in: Search only in specific fields. Allowed values: title, description, author, audio. If not specified, then search every fields
        len_max: Maximum audio length in minutes. Applicable only when type parameter is "episode".
        len_min: Minimum audio length in minutes. Applicable only when type parameter is "episode".
        published_after: Only show episodes published after this timestamp (in milliseconds). Default: 0 (since the very first episode). If published_before & published_after are used at the same time, published_before should be bigger than published_after.
        published_before: Only show episodes published before this timestamp (in milliseconds). Default: now.  If published_before & published_after are used at the same time, published_before should be bigger than published_after.
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/search"
    querystring = {'q': q, }
    if type:
        querystring['type'] = type
    if genre_ids:
        querystring['genre_ids'] = genre_ids
    if language:
        querystring['language'] = language
    if safe_mode:
        querystring['safe_mode'] = safe_mode
    if sort_by_date:
        querystring['sort_by_date'] = sort_by_date
    if offset:
        querystring['offset'] = offset
    if ocid:
        querystring['ocid'] = ocid
    if ncid:
        querystring['ncid'] = ncid
    if only_in:
        querystring['only_in'] = only_in
    if len_max:
        querystring['len_max'] = len_max
    if len_min:
        querystring['len_min'] = len_min
    if published_after:
        querystring['published_after'] = published_after
    if published_before:
        querystring['published_before'] = published_before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_meta_data_for_an_episode_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch meta data for a specific episode. You can get id from search api's response."
    id: id for a specific episode. You can get id from search api's response.
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/episodes/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def typeahead(q: str, safe_mode: int=1, show_genres: int=1, show_podcasts: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Suggest search terms"
    q: Search term
        safe_mode: Whether or not to exclude podcasts/episodes with explicit language. 1 is yes and 0 is no. Default: 0. This is only applicable when show_podcasts=1
        show_genres: Autosuggest genres. 1 is yes, 0 is no. Default: 0
        show_podcasts: Autosuggest podcasts. This only searches podcast title and publisher and returns very limited info of 5 podcasts. 1 is yes, 0 is no. It's a bit slow to autosuggest podcasts, so we turn it off by default. Default: 0. If show_podcasts=1, you can also pass iTunes id (e.g., 474722933) to the q parameter to fetch podcast meta data.
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/typeahead"
    querystring = {'q': q, }
    if safe_mode:
        querystring['safe_mode'] = safe_mode
    if show_genres:
        querystring['show_genres'] = show_genres
    if show_podcasts:
        querystring['show_podcasts'] = show_podcasts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_list_of_supported_languages_for_podcasts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of languages that are supported in Listen Notes database. You can use the language string as query parameter in the search API"
    
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_list_of_podcast_genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of podcast genres that are supported in Listen Notes. The genre id can be passed to Search API and filter results by specific genres. You may want to cache this list of genres on the client side."
    
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_list_of_best_podcasts_by_genre(page: int=2, safe_mode: int=1, genre_id: int=125, region: str='fr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of curated best podcasts by genre. You can get the genre ids from /api/v1/genres endpoint"
    page: Page number of those podcasts in this genre.
        safe_mode: Whether or not to exclude podcasts/episodes with explicit language. 1 is yes and 0 is no. Default: 0.
        genre_id: You can get the id from /api/v1/genres. If not specified, it'll be all podcasts.
        region: Filter best podcasts by country/region. You can get the supported country codes from the /api/v1/regions endpoint. Default: us
        
    """
    url = f"https://listennotes.p.rapidapi.com/api/v1/best_podcasts"
    querystring = {}
    if page:
        querystring['page'] = page
    if safe_mode:
        querystring['safe_mode'] = safe_mode
    if genre_id:
        querystring['genre_id'] = genre_id
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "listennotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

