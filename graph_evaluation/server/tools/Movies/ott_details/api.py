import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def advanced_search(start_year: int=1970, page: int=1, max_imdb: int=7, type: str='movie', sort: str='latest', min_imdb: int=6, genre: str='action', language: str='english', end_year: int=2020, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search for a movie or tvshow based on various parameters such as release year , imdb rating , genre , language etc."
    start_year: Enter any year between 1970 to 2020 to get results.
        type: Enter type 'movie' or 'show'
        sort: Enter values highestrated , lowestrated , latest , oldest to  sort results accodingly .
        genre: use comma seperated values to enter multiple genre eg : action,horror
        language: use comma seperated values to enter multiple values , eg : english,german
        end_year: Enter any year from 1970 to 2020 to  get results.
        
    """
    url = f"https://ott-details.p.rapidapi.com/advancedsearch"
    querystring = {}
    if start_year:
        querystring['start_year'] = start_year
    if page:
        querystring['page'] = page
    if max_imdb:
        querystring['max_imdb'] = max_imdb
    if type:
        querystring['type'] = type
    if sort:
        querystring['sort'] = sort
    if min_imdb:
        querystring['min_imdb'] = min_imdb
    if genre:
        querystring['genre'] = genre
    if language:
        querystring['language'] = language
    if end_year:
        querystring['end_year'] = end_year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basic_info(peopleid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info on a  cast such as name , profession , birth and death year , bio , poster , best titles  etc."
    
    """
    url = f"https://ott-details.p.rapidapi.com/getcastDetails"
    querystring = {'peopleid': peopleid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def params(param: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get array of values that can be used as params in Advanced Search ."
    param: input 'genre' or 'language' to get array of genre or languages that can be used as filter in advanced search .
        
    """
    url = f"https://ott-details.p.rapidapi.com/getParams"
    querystring = {'param': param, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(title: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search for a movie or tvshow based on the given 'title '."
    page: *Maximum number of pages returned is 10 
        
    """
    url = f"https://ott-details.p.rapidapi.com/search"
    querystring = {'title': title, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_details(imdbid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get basic information on a movie or tv show such as imdbid , title , genre , runtime , imdbrating , language , synopsis , type , imageurl  , Streaming platforms info etc."
    
    """
    url = f"https://ott-details.p.rapidapi.com/gettitleDetails"
    querystring = {'imdbid': imdbid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def additional_title_details(imdbid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  additional details for a movie or tv show like reviews  , quotes , plotsummary , number of votes , trailer url  , cast details etc."
    
    """
    url = f"https://ott-details.p.rapidapi.com/getadditionalDetails"
    querystring = {'imdbid': imdbid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ott_providers(region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  information on OTT platforms we suuport ."
    region: currently only USA and India region is supported enter param 'US' for USA and 'IN' for India.
        
    """
    url = f"https://ott-details.p.rapidapi.com/getPlatforms"
    querystring = {'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_arrivals(region: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest arrivals from different platforms ."
    region: Use 'IN' for India and 'US' for USA , * currently we support only US and Indian region.
        
    """
    url = f"https://ott-details.p.rapidapi.com/getnew"
    querystring = {'region': region, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

