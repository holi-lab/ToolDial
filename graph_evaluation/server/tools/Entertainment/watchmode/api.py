import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def title_episodes(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all of the episodes for a TV series or mini-series, as well as the streaming sources each episode is available on."
    id: This ID can be the Watchmode ID of the title (returned by other endpoints such as the list-titles endpoint), or found in the mapping file: https://api.watchmode.com/datasets/title_id_map.csv.

You can also pass an IMDB ID here instead, or a TMDB type and TMDB ID combination. For example, the TMDB combination for The Shawshank Redemption is movie-278 and for Breaking Bad is tv-1396. 
        
    """
    url = f"https://watchmode.p.rapidapi.com/title/{is_id}/episodes/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete_search(search_value: str, search_type: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for titles/and or people by name or a partial name. Useful for building an autocomplete search of titles and/or people. The results include the field result_type to indicate which type of result it is (title or person). For titles, the movie poster will be returned in image_url, for a person a headshot will be returned in image_url."
    search_value: The phrase to search for, can be a full title or person name, or a partial phrase. For example searching for \"The sha\" will find the movie \"The Shawshank Redemption\".
        search_type: Set this to 1 to include titles and people in results. Set this to 2 to include titles only. Set this to 3 to include movies only. Set this to 4 to include TV only. Set this to 5 to include people only. By default this is set to 1.
        
    """
    url = f"https://watchmode.p.rapidapi.com/autocomplete-search/"
    querystring = {'search_value': search_value, }
    if search_type:
        querystring['search_type'] = search_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streaming_release_dates(start_date: int=20220301, limit: int=250, end_date: int=20220312, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing of recently released or coming soon releases on the major streaming services. Only major services and US releases dates included, however most of the major services (Netflix, Hulu, etc) release original content on the same days in all countries they support. We return is_original field to indicate wheter the title is an original release on that streaming service."
    start_date: By default, this endpoint will return release dates from the current date through the next 30 days. Populate this parameter to manually set a start date to include releases from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        limit: Set how many release dates to return, default is 500.
        end_date: By default, this endpoint will return release dates from the current date through the next 30 days. Populate this parameter to manually set a end date to include releases from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        
    """
    url = f"https://watchmode.p.rapidapi.com/releases"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if limit:
        querystring['limit'] = limit
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(search_field: str, search_value: str, types: str='tv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for titles or people using an external ID (IMDB, TheMovieDB.org), or by name. Returns an array of results objects, that can either be a title or a person. Useful for getting the Watchmode IDs for titles and people. For example, you can set the parameters to search_value=Breaking%20Bad and search_field=name to get all of the titles named "Breaking bad", and then use the IDs returned in other endpoints such as /v1/title/"
    search_field: The field for us to search in, either a 3rd party ID or \\\"name\\\" which will search for a movie/show title or a person's name depending on the type(s) set. Must be one of the following options:
imdb_id, tmdb_person_id, tmdb_movie_id, tmdb_tv_id, name.
        search_value: The value we should search for. For example, if you set search_field to imdb_id, this would be the IMDB title/person ID, eg. tt0944947.
        types: Pass one of the following values, or multiple comma separated values to only return certain types:
tv, movie, person.
        
    """
    url = f"https://watchmode.p.rapidapi.com/search/"
    querystring = {'search_field': search_field, 'search_value': search_value, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return details on a specific person (actor, director, etc)."
    
    """
    url = f"https://watchmode.p.rapidapi.com/person/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_titles(genres: str='4,9', limit: int=250, source_ids: str='23,206', source_types: str='sub,free', types: str='movie,tv_series', regions: str='US', sort_by: str='relevance_desc', page: int=1, network_ids: str='1,8,12', release_date_start: int=20010101, release_date_end: int=20201211, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing of titles that match certain parameters. This powerful endpoint can allow you to find many combinations of titles. For example you could search for something as granular "Horror Movies Streaming on Netflix in the USA" by using the genres, types, source_ids and regions parameters.
		
		Results are paginated, and return 250 pages per query by default. Useful for mapping all Watchmode title IDs in your app, and finding in bulk what titles are available in different countries, different sources or source types.
		
		Streaming sources are limited to USA only for free plans."
    genres: Filter results to only include certain genre(s). Pass in a single genre id (which you would get from the /v1/genres/ endpoint, or multiple comma separated.
        limit: Set how many titles to return per page, default and maximum is 250.
        source_ids: Pass an individual ID for a source (returned from the /sources/ endpoint) to filter the results to titles available on that source. Pass multiple values comma separated to return titles available on one of the sources you pass in.
Note: If you populate this you can only set a single region, and if you set no region US will be set by default.
        source_types: Filter results to only include titles that are available on a specific type(s) of source (such a subscription, or TV Everywhere channel apps, etc). By default all are selected, pass one or multiple (comma delimited) of these values: sub, rent, buy, free, tve
Note: If you populate this you can only set a single region, and if you set no region US will be set by default.
        types: Filter result to only include certain types of titles. Pass a single type or pass multiple types comma delimited. Possible values: movie, tv_series, tv_special, tv_miniseries, short_film
        regions: Pass one of the region values (eg. US), or multiple regions comma delimited to only return sources active in those regions. Currently supported regions: US, GB, CA, AU
Note: If you populate the source_ids or source_types you can only set a single region, and if you set no region US will be set by default. 
        sort_by: Sort order of results, possible values: relevance_desc, relevance_asc, popularity_desc, popularity_asc, release_date_desc, release_date_asc, title_desc, title_asc. Default value is: relevance_desc.
        page: Set the page of results you want to return, if this isn't set you will always get page 1 returned.
        network_ids: Pass an individual ID for a TV network (returned from the /networks/ endpoint) to filter the results to titles the originally aired on that TV network. Pass multiple values comma separated to return titles that aired on one of the networks you passed in.
        release_date_start: Set the start of a range for when the title was released (the initial release of the movie or show, not necessarily when it was released on a streaming service). For example, to only include releases on or after January 1, 2001 set this to 20010101
        release_date_end: Set the end of a range for when the title was released (the initial release of the movie or show, not necessarily when it was released on a streaming service). For example, to only include releases on or before December 11, 2020 set this to 20201211
        
    """
    url = f"https://watchmode.p.rapidapi.com/list-titles/"
    querystring = {}
    if genres:
        querystring['genres'] = genres
    if limit:
        querystring['limit'] = limit
    if source_ids:
        querystring['source_ids'] = source_ids
    if source_types:
        querystring['source_types'] = source_types
    if types:
        querystring['types'] = types
    if regions:
        querystring['regions'] = regions
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if network_ids:
        querystring['network_ids'] = network_ids
    if release_date_start:
        querystring['release_date_start'] = release_date_start
    if release_date_end:
        querystring['release_date_end'] = release_date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a listing of all regions (countries) that Watchmode currently supports and their 2 letter country codes used in the return data of other endpoints."
    
    """
    url = f"https://watchmode.p.rapidapi.com/regions/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sources(regions: str='US,CA', types: str='sub,free', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a listing of all streaming sources that Watchmode supports. Optionally filter by type of source (subscription, free, etc)."
    regions: Pass one of the region values (eg. US), or multiple regions comma delimited to only return sources active in those regions. Currently supported regions: US, GB, CA, AU
        types: Pass one of the following values, or multiple comma separated values to only return certain types of streaming sources:
sub, free, purchase, tve. \\\"sub\\\" means the service is a subscription service, \\\"tve\\\" means the service is a TV channel app.
        
    """
    url = f"https://watchmode.p.rapidapi.com/sources/"
    querystring = {}
    if regions:
        querystring['regions'] = regions
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a mapping of genre names and IDs. Some genres have a tmdb_id, which is the corresponding genre ID on TheMovieDB.org API."
    
    """
    url = f"https://watchmode.p.rapidapi.com/genres/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_titles(page: int=1, types: str='movie,tv_series', limit: int=50, end_date: int=None, start_date: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing of the title IDs of new titles added to Watchmode within the date range. You can use this to find new titles, then use the /v1/title/ endpoint to get details on the title."
    page: Set the page of results you want to return, if this isn't set you will always get page 1 returned.
        types: Filter result to only include certain types of titles. Pass a single type or pass multiple types comma delimited. Possible values: movie, tv_series, tv_special, tv_miniseries, short_film
        limit: Set how many titles to return per page, default and maximum is 250.
        end_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a end date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021).
        start_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a start date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021).
        
    """
    url = f"https://watchmode.p.rapidapi.com/changes/new_titles/"
    querystring = {}
    if page:
        querystring['page'] = page
    if types:
        querystring['types'] = types
    if limit:
        querystring['limit'] = limit
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def networks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a listing of all TV networks that may be returned for a title in the /title endpoint."
    
    """
    url = f"https://watchmode.p.rapidapi.com/networks/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_details(is_id: str, language: str='ES', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific title, using the Watchmode universal ID."
    id: This ID can be the Watchmode ID of the title (returned by other endpoints such as the list-titles endpoint), or found in the mapping file: https://api.watchmode.com/datasets/title_id_map.csv.

You can also pass an IMDB ID here instead, or a TMDB type and TMDB ID combination. For example, the TMDB combination for The Shawshank Redemption is movie-278 and for Breaking Bad is tv-1396. 
        language: Two letter iso_639_1 language code. Return the title and plot overview in the language of your choosing (default is EN). If this is set, and not to EN, then an additional field called english_title will be returned with the title in English if available.

        
    """
    url = f"https://watchmode.p.rapidapi.com/title/{is_id}/details/"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_seasons(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all of the seasons for a TV show or mini-series."
    id: This ID can be the Watchmode ID of the title (returned by other endpoints such as the list-titles endpoint), or found in the mapping file: https://api.watchmode.com/datasets/title_id_map.csv.

You can also pass an IMDB ID here instead, or a TMDB type and TMDB ID combination. For example, the TMDB combination for The Shawshank Redemption is movie-278 and for Breaking Bad is tv-1396. 
        
    """
    url = f"https://watchmode.p.rapidapi.com/title/{is_id}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_streaming_sources(is_id: str, regions: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the streaming sources this title is available on and direct links to watch the title (web, iOS deeplinks and Android deeplinks for paid users). Streaming sources are limited to USA only for free plans."
    id: This ID can be the Watchmode ID of the title (returned by other endpoints such as the list-titles endpoint), or found in the mapping file: https://api.watchmode.com/datasets/title_id_map.csv.

You can also pass an IMDB ID here instead, or a TMDB type and TMDB ID combination. For example, the TMDB combination for The Shawshank Redemption is movie-278 and for Breaking Bad is tv-1396. 
        regions: Pass one of the 2 character country codes from the /regions/ endpoint to filter the streaming source results to certain countries. By default the API will return all regions. Pass multiple countries by submitting them comma separated.

        
    """
    url = f"https://watchmode.p.rapidapi.com/title/{is_id}/sources/"
    querystring = {}
    if regions:
        querystring['regions'] = regions
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_cast_crew(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all people associated with the title, the "cast" (actors), and "crew" (directors, writers, cinematographers, etc). For more details on a person, pass the person-id to the /person/ endpoint."
    id: This ID can be the Watchmode ID of the title (returned by other endpoints such as the list-titles endpoint), or found in the mapping file: https://api.watchmode.com/datasets/title_id_map.csv.

You can also pass an IMDB ID here instead, or a TMDB type and TMDB ID combination. For example, the TMDB combination for The Shawshank Redemption is movie-278 and for Breaking Bad is tv-1396. 
        
    """
    url = f"https://watchmode.p.rapidapi.com/title/{is_id}/cast-crew/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_with_changed_sources(start_date: int=None, limit: int=50, end_date: int=None, regions: str='US,CA', types: str='movie,tv_series', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing of titles that have changed to their streaming sources within the date range."
    start_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a start date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        limit: Set how many titles to return per page, default and maximum is 250.
        end_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a end date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        regions: Pass in the 2 character region code (eg US) for the country you want to get titles with changed sources from. There is a limit to 1 region on this endpoint, if you leave this field blank US changes will be returned only.
        types: Filter result to only include certain types of titles. Pass a single type or pass multiple types comma delimited. Possible values: movie, tv_series, tv_special, tv_miniseries, short_film
        page: Set the page of results you want to return, if this isn't set you will always get page 1 returned.
        
    """
    url = f"https://watchmode.p.rapidapi.com/changes/titles_sources_changed/"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if limit:
        querystring['limit'] = limit
    if end_date:
        querystring['end_date'] = end_date
    if regions:
        querystring['regions'] = regions
    if types:
        querystring['types'] = types
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_people(page: int=1, start_date: int=None, limit: int=50, end_date: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing of the ids of new people added to Watchmode within the date range."
    page: Set the page of results you want to return, if this isn't set you will always get page 1 returned.
        start_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a start date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        limit: Set how many titles to return per page, default and maximum is 250.
        end_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a end date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        
    """
    url = f"https://watchmode.p.rapidapi.com/changes/new_people/"
    querystring = {}
    if page:
        querystring['page'] = page
    if start_date:
        querystring['start_date'] = start_date
    if limit:
        querystring['limit'] = limit
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_with_changed_episodes(end_date: str=None, start_date: str=None, page: int=1, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing of titles that have changes to their episodes (new episodes, episode details changed, etc) within the date range."
    end_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a end date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        start_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a start date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        page: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a end date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021).
        limit: Set the page of results you want to return, if this isn't set you will always get page 1 returned.
        
    """
    url = f"https://watchmode.p.rapidapi.com/changes/titles_episodes_changed/"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_with_changed_details(end_date: int=None, page: int=1, limit: int=50, types: str='movie,tv_series', start_date: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing of titles that have had a change to their basic details (overview, cast, genres, ratings, etc) within the date range."
    end_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a end date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        page: Set the page of results you want to return, if this isn't set you will always get page 1 returned.
        limit: Set how many titles to return per page, default and maximum is 250.
        types: Filter result to only include certain types of titles. Pass a single type or pass multiple types comma delimited. Possible values: movie, tv_series, tv_special, tv_miniseries, short_film
        start_date: By default, the changes API will return changes since the beginning of yesterday. Populate this parameter to manually set a start date to include changes from a specific date range, format is YYYYMMDD (eg 20210101 for January 1st, 2021), or for more granular results down to the second YYYYMMDDHHMMSS (eg 20210101123459 for January 1st, 2021 12:24pm and 59 seconds). Hours are in 24 hour format and in Eastern Standard Time. Month, day, hour, minutes and seconds use leading zeros for single digit values.
        
    """
    url = f"https://watchmode.p.rapidapi.com/changes/titles_details_changed/"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if types:
        querystring['types'] = types
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "watchmode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

