import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_artist_audience_details(source: str, country_code: str, songstats_artist_id: str='vxk62ige', spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed audience information of an artist for a specific country filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        country_code: A two-letter country code
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/audience/details"
    querystring = {'source': source, 'country_code': country_code, }
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_label(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a record label."
    q: Search Query
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_top_playlists(source: str, rank_type: str='playlists_current', limit: int=20, current: bool=None, beatport_label_id: str='71740', songstats_label_id: str='q754od0s', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top playlists of a label."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        rank_type: The rank type to sort top tracks by (can be any available_rank_type for a given source).
        limit: Specifies the limit of results being returned.
        beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/top_playlists"
    querystring = {'source': source, }
    if rank_type:
        querystring['rank_type'] = rank_type
    if limit:
        querystring['limit'] = limit
    if current:
        querystring['current'] = current
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_top_tracks(beatport_label_id: str='71740', songstats_label_id: str='q754od0s', rank_type: str='playlists_current', limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top tracks of a label sorted by rank type."
    beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        rank_type: The rank type to sort top tracks by (can be any available_rank_type for a given source).
        limit: Specifies the limit of results being returned.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/top_tracks"
    querystring = {}
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    if rank_type:
        querystring['rank_type'] = rank_type
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_current_stats(source: str, beatport_label_id: str='71740', songstats_label_id: str='q754od0s', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns summary and detailed information of a record label filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/stats"
    querystring = {'source': source, }
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_activities(beatport_label_id: str='74932', source: str='spotify,apple_music,deezer', limit: int=20, tier: int=4, songstats_label_id: str='7gk4yfc9', recent: bool=None, milestones: bool=None, albums: bool=None, editorial: bool=None, offset: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the latest activities of a label filtered by source."
    beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        source: Specifies the sources that data is being requested for. Can be a comma separated list of individual sources or be set to 'all'.
        limit: Specifies the limit of results being returned.
        tier: Filter Activities by importance. Tier 4 is the lowest tier that returns all activities. Tier 1 is the highest tier that returns only the most significant activities.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        recent: Only include Activities for recent tracks (released in past 3 months).
        milestones: Include Milestone Activities.
        albums: Include Album Activities.
        editorial: Only include Editorial Activities.
        offset: Specifies the offset of the limited results being returned.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/activities"
    querystring = {}
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if source:
        querystring['source'] = source
    if limit:
        querystring['limit'] = limit
    if tier:
        querystring['tier'] = tier
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    if recent:
        querystring['recent'] = recent
    if milestones:
        querystring['milestones'] = milestones
    if albums:
        querystring['albums'] = albums
    if editorial:
        querystring['editorial'] = editorial
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_audience_details(source: str, country_code: str, beatport_label_id: str='71740', songstats_label_id: str='q754od0s', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed audience information of a record label for a specific country filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        country_code: A two-letter country code
        beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/audience/details"
    querystring = {'source': source, 'country_code': country_code, }
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_artist(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for an artist."
    q: Search Query
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_top_playlists(source: str, spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', songstats_artist_id: str='vxk62ige', limit: int=20, current: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top playlists of an artist."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        limit: Specifies the limit of results being returned.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/top_playlists"
    querystring = {'source': source, }
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    if limit:
        querystring['limit'] = limit
    if current:
        querystring['current'] = current
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_catalog(limit: int=20, spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', offset: int=20, songstats_artist_id: str='vxk62ige', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the track catalog of an artist."
    limit: Specifies the limit of results being returned.
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        offset: Specifies the offset of the limited results being returned.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/catalog"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    if offset:
        querystring['offset'] = offset
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_audience(source: str, songstats_artist_id: str='vxk62ige', spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns audience information of an artist filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/audience"
    querystring = {'source': source, }
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_historic_stats(source: str, with_aggregates: bool=None, end_date: str='2022-01-16', songstats_artist_id: str='vxk62ige', spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', start_date: str='2021-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns historic information of an artist filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        with_aggregates: Includes data points aggregated by Songstats like playlists_current, playlists_reach_current, charts_current, etc.
        end_date: Specifies the end_date of the historic data that is being requested.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        start_date: Specifies the start_date of the historic data that is being requested.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/historic_stats"
    querystring = {'source': source, }
    if with_aggregates:
        querystring['with_aggregates'] = with_aggregates
    if end_date:
        querystring['end_date'] = end_date
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_activities(source: str='spotify,apple_music,deezer', songstats_artist_id: str='vxk62ige', offset: int=20, limit: int=20, spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', tier: int=4, albums: bool=None, milestones: bool=None, editorial: bool=None, recent: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the latest activities of an artist filtered by source."
    source: Specifies the sources that data is being requested for. Can be a comma separated list of individual sources or be set to 'all'.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        offset: Specifies the offset of the limited results being returned.
        limit: Specifies the limit of results being returned.
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        tier: Filter Activities by importance. Tier 4 is the lowest tier that returns all activities. Tier 1 is the highest tier that returns only the most significant activities.
        albums: Include Album Activities.
        milestones: Include Milestone Activities.
        editorial: Only include Editorial Activities.
        recent: Only include Activities for recent tracks (released in past 3 months).
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/activities"
    querystring = {}
    if source:
        querystring['source'] = source
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    if tier:
        querystring['tier'] = tier
    if albums:
        querystring['albums'] = albums
    if milestones:
        querystring['milestones'] = milestones
    if editorial:
        querystring['editorial'] = editorial
    if recent:
        querystring['recent'] = recent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_info(spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', songstats_artist_id: str='vxk62ige', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns metadata information of an artist."
    spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/info"
    querystring = {}
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_audience(source: str, beatport_label_id: str='71740', songstats_label_id: str='q754od0s', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns audience information of a record label filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/audience"
    querystring = {'source': source, }
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_catalog(beatport_label_id: str='71740', songstats_label_id: str='q754od0s', limit: int=20, offset: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the track catalog of a record label."
    beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        limit: Specifies the limit of results being returned.
        offset: Specifies the offset of the limited results being returned.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/catalog"
    querystring = {}
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_historic_label_stats(source: str, start_date: str='2021-12-31', end_date: str='2022-01-16', songstats_label_id: str='q754od0s', beatport_label_id: str='71740', with_aggregates: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns historic information of a record label filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        start_date: Specifies the start_date of the historic data that is being requested.
        end_date: Specifies the end_date of the historic data that is being requested.
        songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        with_aggregates: Includes data points aggregated by Songstats like playlists_current, playlists_reach_current, charts_current, etc.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/historic_stats"
    querystring = {'source': source, }
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    if with_aggregates:
        querystring['with_aggregates'] = with_aggregates
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_label_info(songstats_label_id: str='q754od0s', beatport_label_id: str='71740', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns metadata information of a record label."
    songstats_label_id: The Songstats ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        beatport_label_id: The Beatport ID of the label being requested *

*Either `beatport_label_id` or `songstats_label_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/labels/info"
    querystring = {}
    if songstats_label_id:
        querystring['songstats_label_id'] = songstats_label_id
    if beatport_label_id:
        querystring['beatport_label_id'] = beatport_label_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_top_tracks(rank_type: str='playlists_current', spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', limit: int=20, songstats_artist_id: str='vxk62ige', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top tracks of an artist sorted rank type."
    rank_type: The rank type to sort top tracks by (can be any available_rank_type for a given source).
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        limit: Specifies the limit of results being returned.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/top_tracks"
    querystring = {}
    if rank_type:
        querystring['rank_type'] = rank_type
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    if limit:
        querystring['limit'] = limit
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_current_stats(source: str, spotify_artist_id: str='2h93pZq0e7k5yf4dywlkpM', songstats_artist_id: str='vxk62ige', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns summary and detailed information of an artist filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        spotify_artist_id: The Spotify ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        songstats_artist_id: The Songstats ID of the artist being requested *

*Either `spotify_artist_id` or `songstats_artist_id` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/artists/stats"
    querystring = {'source': source, }
    if spotify_artist_id:
        querystring['spotify_artist_id'] = spotify_artist_id
    if songstats_artist_id:
        querystring['songstats_artist_id'] = songstats_artist_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_track(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a track."
    
    """
    url = f"https://songstats.p.rapidapi.com/tracks/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_current_stats(source: str, spotify_track_id: str='3VTPi12rK7mioSLL0mmu75', isrc: str='GBKQU2166610', with_playlists: bool=None, with_charts: bool=None, with_links: bool=None, songstats_track_id: str='2iye8blt', with_videos: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns summary and detailed information of a track filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        spotify_track_id: The Spotify ID of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        isrc: The ISRC of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        with_playlists: Includes an array of all of the playlists associated to the track.
        with_charts: Includes an array of all of the charts associated to the track.
        with_links: Includes an array of all of the source links associated to the track.
        songstats_track_id: The Songstats ID of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        with_videos: Includes an array of all of the videos currently associated to the track.
        
    """
    url = f"https://songstats.p.rapidapi.com/tracks/stats"
    querystring = {'source': source, }
    if spotify_track_id:
        querystring['spotify_track_id'] = spotify_track_id
    if isrc:
        querystring['isrc'] = isrc
    if with_playlists:
        querystring['with_playlists'] = with_playlists
    if with_charts:
        querystring['with_charts'] = with_charts
    if with_links:
        querystring['with_links'] = with_links
    if songstats_track_id:
        querystring['songstats_track_id'] = songstats_track_id
    if with_videos:
        querystring['with_videos'] = with_videos
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_info(spotify_track_id: str='3VTPi12rK7mioSLL0mmu75', songstats_track_id: str='2iye8blt', isrc: str='GBKQU2166610', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns metadata information of a track."
    spotify_track_id: The Spotify ID of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        songstats_track_id: The Songstats ID of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        isrc: The ISRC of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/tracks/info"
    querystring = {}
    if spotify_track_id:
        querystring['spotify_track_id'] = spotify_track_id
    if songstats_track_id:
        querystring['songstats_track_id'] = songstats_track_id
    if isrc:
        querystring['isrc'] = isrc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns currently supported data sources."
    
    """
    url = f"https://songstats.p.rapidapi.com/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_historic_stats(source: str, start_date: str='2021-12-31', songstats_track_id: str='2iye8blt', spotify_track_id: str='3VTPi12rK7mioSLL0mmu75', with_aggregates: bool=None, end_date: str='2022-01-16', isrc: str='GBKQU2166610', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns historic information of a track filtered by source."
    source: Specifies the sources that data is being requested for *

*Can be a comma separated list of individual sources like `spotify,apple_music,deezer` or be set to `all`.
        start_date: Specifies the start_date of the historic data that is being requested.
        songstats_track_id: The Songstats ID of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        spotify_track_id: The Spotify ID of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        with_aggregates: Includes data points aggregated by Songstats like playlists_current, playlists_reach_current, charts_current, etc.
        end_date: Specifies the end_date of the historic data that is being requested.
        isrc: The ISRC of the track being requested *

*Either `spotify_track_id`,  `songstats_track_id` or `isrc` **are required** for this request.
        
    """
    url = f"https://songstats.p.rapidapi.com/tracks/historic_stats"
    querystring = {'source': source, }
    if start_date:
        querystring['start_date'] = start_date
    if songstats_track_id:
        querystring['songstats_track_id'] = songstats_track_id
    if spotify_track_id:
        querystring['spotify_track_id'] = spotify_track_id
    if with_aggregates:
        querystring['with_aggregates'] = with_aggregates
    if end_date:
        querystring['end_date'] = end_date
    if isrc:
        querystring['isrc'] = isrc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songstats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

