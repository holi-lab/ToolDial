import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Netflix genres"
    
    """
    url = f"https://unogsng.p.rapidapi.com/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of available Netflix countries (includes uNoGS country id)"
    
    """
    url = f"https://unogsng.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(netflixid: int, offset: int=3, limit: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pull all the images associated with a particular title"
    netflixid: Netflix Title ID
        offset: Starting Number of results (Default is 0)
        limit: Limit of returned items default/max 100
        
    """
    url = f"https://unogsng.p.rapidapi.com/images"
    querystring = {'netflixid': netflixid, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_countries(netflixid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all countries associated with a particular Netflix ID"
    netflixid: Netflix ID
        
    """
    url = f"https://unogsng.p.rapidapi.com/titlecountries"
    querystring = {'netflixid': netflixid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_genres(netflixid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all genres associated with a particular Netflix ID"
    netflixid: Netflix ID
        
    """
    url = f"https://unogsng.p.rapidapi.com/titlegenres"
    querystring = {'netflixid': netflixid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_details(netflixid: int=81043135, imdbid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Specific information for a given Netflix title"
    netflixid: A Netflix specific ID
        imdbid: An IMDB ID
        
    """
    url = f"https://unogsng.p.rapidapi.com/title"
    querystring = {}
    if netflixid:
        querystring['netflixid'] = netflixid
    if imdbid:
        querystring['imdbid'] = imdbid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def expiring(countrylist: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of expiring titles"
    countrylist: List of uNoGS country IDs separated by a ,
        offset: Starting Number of results (Default is 0)
        limit: Limit of returned items default/max 100
        
    """
    url = f"https://unogsng.p.rapidapi.com/expiring"
    querystring = {'countrylist': countrylist, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def people(offset: int=None, name: str='Brad Pitt', netflixid: int=70021641, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Netflix People by name (returns person id for searches)"
    offset: Starting Number of results (Default is 0)
        name: A persons name First Last or Both
        netflixid: A Netflix title ID
        limit: Limit of returned items default/max 100
        
    """
    url = f"https://unogsng.p.rapidapi.com/people"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    if netflixid:
        querystring['netflixid'] = netflixid
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def episodes(seasonid: int=70051768, episodeid: int=70150654, netflixid: int=70153392, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Netflix People by name (returns person id for searches)"
    netflixid: A Netflix title ID
        
    """
    url = f"https://unogsng.p.rapidapi.com/episodes"
    querystring = {}
    if seasonid:
        querystring['seasonid'] = seasonid
    if episodeid:
        querystring['episodeid'] = episodeid
    if netflixid:
        querystring['netflixid'] = netflixid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deleted(offset: str='0', netflixid: str=None, limit: str=None, date: str='2019-12-01', countrylist: str='78', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all title which have been deleted meeting a limited criteria"
    netflixid: Netflix Title ID
        date: returns anything deleted after this date
        countrylist: list of uNoGS country ID's separated by a ,
        
    """
    url = f"https://unogsng.p.rapidapi.com/titlesdel"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if netflixid:
        querystring['netflixid'] = netflixid
    if limit:
        querystring['limit'] = limit
    if date:
        querystring['date'] = date
    if countrylist:
        querystring['countrylist'] = countrylist
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(start_year: int=1972, type: str=None, audio: str='english', audiosubtitle_andor: str='and', limit: int=100, end_rating: int=None, query: str=None, countrylist: str='78,46', country_andorunique: str='unique', orderby: str='rating', newdate: str=None, genrelist: str=None, start_rating: str=None, end_year: int=2019, offset: int=0, subtitle: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Netflix titles based on a variety of parameters"
    start_year: A 4 digit year
        type: movie or series
        audio: A valid language type
        audiosubtitle_andor: and/or for audio and subtitles
        limit: Limit of returned items default/max 100
        end_rating: imdb rating 0-10
        query: any string you want to search (fulltext against the title)
        countrylist: list of uNoGS country ID's (from country endpoint) leave blank for all country search
        orderby: orderby string (date,dateDesc,rating,title,type,runtime,filmyear,filmyearDesc)
        newdate: returns all titles with a new date greater than this
        genrelist: list of Netflix genre id's (see genre endpoint for list)
        start_rating: imdb rating 0-10
        end_year: A 4 digit year
        offset: Starting Number of results (Default is 0)
        subtitle: A valid language type
        
    """
    url = f"https://unogsng.p.rapidapi.com/search"
    querystring = {}
    if start_year:
        querystring['start_year'] = start_year
    if type:
        querystring['type'] = type
    if audio:
        querystring['audio'] = audio
    if audiosubtitle_andor:
        querystring['audiosubtitle_andor'] = audiosubtitle_andor
    if limit:
        querystring['limit'] = limit
    if end_rating:
        querystring['end_rating'] = end_rating
    if query:
        querystring['query'] = query
    if countrylist:
        querystring['countrylist'] = countrylist
    if country_andorunique:
        querystring['country_andorunique'] = country_andorunique
    if orderby:
        querystring['orderby'] = orderby
    if newdate:
        querystring['newdate'] = newdate
    if genrelist:
        querystring['genrelist'] = genrelist
    if start_rating:
        querystring['start_rating'] = start_rating
    if end_year:
        querystring['end_year'] = end_year
    if offset:
        querystring['offset'] = offset
    if subtitle:
        querystring['subtitle'] = subtitle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogsng.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

