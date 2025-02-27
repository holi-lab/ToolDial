import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_songs_with_all_metadata_and_stats_play_counts(offset: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all tracks with full metadata including recent Spotify play counts.
		
		Also Apple song URL, Spotify song URL, Composer name, ISRC, Label, Release date, Genres, Apple IDs, Spotify IDs are included."
    
    """
    url = f"https://indie-songs-distrokid-unsigned.p.rapidapi.com/tracks/full"
    querystring = {'offset': offset, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indie-songs-distrokid-unsigned.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_50_indie_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get TOP 50 indie songs based on their daily stream increase ratio"
    
    """
    url = f"https://indie-songs-distrokid-unsigned.p.rapidapi.com/tracks/top50/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indie-songs-distrokid-unsigned.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_indie_songs_by_filters(by: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search song by track name,  artist name, label, release date, ISRC, composer name, genre"
    by: 'trackname', 'artistname', 'isrc', 'label',  'genre', 'composer_name', or 'release_date'
        
    """
    url = f"https://indie-songs-distrokid-unsigned.p.rapidapi.com/tracks/search2"
    querystring = {'by': by, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indie-songs-distrokid-unsigned.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_csv_of_top_50_indie_songs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get TOP 50 indie songs by their daily stream increase in a CSV format using URL"
    
    """
    url = f"https://indie-songs-distrokid-unsigned.p.rapidapi.com/tracks/top50/csv2"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indie-songs-distrokid-unsigned.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

