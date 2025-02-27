import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shows_id_episodes(platform: str, is_id: int, offset: int=0, region: str='US', limit: int=25, sort: str='regular', season: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the episodes from the show"
    platform: Retrieve sources from the specified platform.
*required, possible values: ios, android, androidtv, web*
        id: The id of the show
        offset: Skips this number of records.
*optional, default: 0*
        region: Filter by region.
*optional, default: US*
*possible values: US, FR*

        limit: Number of records to return per request.
*optional, default: 25, max: 100*
        sort: Sort the records in regular order or reverse order to get most recent episodes first.
*optional, default: regular*
*possible values: regular, reverse*
        season: A particular season for a show. Do not specify any value to get the episodes from all the seasons.
*optional*
        
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/shows/{is_id}/episodes"
    querystring = {'platform': platform, }
    if offset:
        querystring['offset'] = offset
    if region:
        querystring['region'] = region
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the list of regions covered by the API"
    
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(type: str, query: str, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of titles that match the query."
    type: The type of content of your research.
*required, possible values: movie, show*
        query: The query itself

        limit: Maximum number of record to return.
*optional, default: 10, max: 100*
        
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/search"
    querystring = {'type': type, 'query': query, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shows_id(platform: str, is_id: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the metadata associated with the specified show."
    platform: Retrieve sources from the specified platform.
*required, possible values: ios, android, androidtv, web*
        id: The id of the show. You can also use a tmdb id.
        region: Retrieve the sources' deep link from this region
*optional, default: US*
*possible values: US, FR*
        
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/shows/{is_id}"
    querystring = {'platform': platform, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies_id(platform: str, is_id: int, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the metadata associated with the specified movie."
    platform: Retrieve sources from the specified platform.
*required, possible values: ios, android, androidtv, web*
        id: The id of the movie. You can also use a tmdb id.
        region: Retrieve the sources' deep link from this region
*optional, default: US*
*possible values: US, FR*
        
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/movies/{is_id}"
    querystring = {'platform': platform, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies(limit: int=5, sources: str='netflix,hulu', region: str='US', offset: int=0, sort: str='popularity', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the basic metadata for all movies available for playback in a specific region. The results are ordered by popularity or alphabetical order. You may get additional information about each movie using its ID."
    limit: Number of records to return per request.
*optional, default: 25, max: 100*
        sources: Filter records by source. Multiple sources may be comma-separated.
*optional*
*possible values: free, tv_everywhere, subscription, purchase, a specific source such as netflix or hulu*
        region: Filter by region.
*optional, default: US*
*possible values: US, FR*

        offset: Skips this number of records.
*optional, default: 0*
        sort: Method to sort the records.
*optional, default: alphabetical*
*possible values:alphabetical, popularity*

        
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/movies"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sources:
        querystring['sources'] = sources
    if region:
        querystring['region'] = region
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sources(region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all sources covered in a specific region"
    region: Filter by region.
*required, possible values: US, FR*
        
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/sources"
    querystring = {'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of all genres"
    
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shows(offset: int=0, limit: int=5, sort: str='popularity', region: str='US', sources: str='netflix,hulu', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the basic metadata for all shows available for playback in a specific region. The results are ordered by popularity or alphabetical order. You may get additional information about each show using its ID."
    offset: Skips this number of records.
*optional, default: 0*
        limit: Number of records to return per request.
*optional, default: 25, max: 100*
        sort: Method to sort the records.
*optional, default: alphabetical*
*possible values:alphabetical, popularity*

        region: Filter by region.
*optional, default: US*
*possible values: US, FR*

        sources: Filter records by source. Multiple sources may be comma-separated.
*optional*
*possible values: free, tv_everywhere, subscription, purchase, a specific source such as netflix or hulu*
        
    """
    url = f"https://streamlinewatch-streaming-guide.p.rapidapi.com/shows"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if region:
        querystring['region'] = region
    if sources:
        querystring['sources'] = sources
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streamlinewatch-streaming-guide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

