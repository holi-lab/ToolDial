import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news"
    
    """
    url = f"https://flixster.p.rapidapi.com/news/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(emsid: str, limit: int=20, withcomments: bool=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews related to a movie"
    emsid: The value of emsId field returned in .../movies/get-opening , .../movies/get-popularity , .../movies/get-upcoming , .../movies/get-dvds , .../search endpoint
        limit: The number of items per response, for paging purpose
        withcomments: Whether or not including children comments
        offset: The offset, for paging purpose
        
    """
    url = f"https://flixster.p.rapidapi.com/reviews/list"
    querystring = {'emsId': emsid, }
    if limit:
        querystring['limit'] = limit
    if withcomments:
        querystring['withComments'] = withcomments
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of an actor/actress"
    id: The value of people -> id field returned in .../search endpoint
        
    """
    url = f"https://flixster.p.rapidapi.com/actors/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def theaters_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get theater and showtime information"
    id: The value of id field returned in .../theaters/list or .../search endpoint
        
    """
    url = f"https://flixster.p.rapidapi.com/theaters/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def theaters_list(longitude: int=None, zipcode: str='90002', latitude: int=None, radius: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List theaters around a postal code or GEO location"
    longitude: The GEO longitude
        zipcode: The postal code
        latitude: The GEO latitude
        radius: The radius
        
    """
    url = f"https://flixster.p.rapidapi.com/theaters/list"
    querystring = {}
    if longitude:
        querystring['longitude'] = longitude
    if zipcode:
        querystring['zipCode'] = zipcode
    if latitude:
        querystring['latitude'] = latitude
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies_detail(emsversionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get movie detail"
    emsversionid: The value of emsVersionId field returned in .../movies/get-opening , .../movies/get-popularity , .../movies/get-upcoming , .../movies/get-dvds , .../search endpoint
        
    """
    url = f"https://flixster.p.rapidapi.com/movies/detail"
    querystring = {'emsVersionId': emsversionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies_get_upcoming(limit: int=100, countryid: str='usa', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get upcoming movies"
    limit: The number of itemed returned, maximum is 100
        countryid: One of the following : afg|alb|dza|asm|and|ago|aia|ata|atg|arg|arm|abw|aus|aut|aze|bhs|bhr|bgd|brb|blr|bel|blz|ben|bmu|btn|bol|bes|bih|bwa|bvt|bra|iot|brn|bgr|bfa|bdi|cpv|khm|cmr|can|cym|caf|tcd|chl|chn|cxr|cck|col|com|cod|cog|cok|cri|hrv|cub|cuw|cyp|cze|civ|dnk|dji|dma|dom|ecu|egy|slv|gnq|eri|est|swz|eth|flk|fro|fji|fin|fra|guf|pyf|atf|gab|gmb|geo|deu|gha|gib|grc|grl|grd|glp|gum|gtm|ggy|gin|gnb|guy|hti|hmd|vat|hnd|hkg|hun|isl|ind|idn|irn|irq|irl|imn|isr|ita|jam|jpn|jey|jor|kaz|ken|kir|prk|kor|kwt|kgz|lao|lva|lbn|lso|lbr|lby|lie|ltu|lux|mac|mdg|mwi|mys|mdv|mli|mlt|mhl|mtq|mrt|mus|myt|mex|fsm|mda|mco|mng|mne|msr|mar|moz|mmr|nam|nru|npl|nld|ncl|nzl|nic|ner|nga|niu|nfk|mnp|nor|omn|pak|plw|pse|pan|png|pry|per|phl|pcn|pol|prt|pri|qat|mkd|rou|rus|rwa|reu|blm|shn|kna|lca|maf|spm|vct|wsm|smr|stp|sau|sen|srb|syc|sle|sgp|sxm|svk|svn|slb|som|zaf|sgs|ssd|esp|lka|sdn|sur|sjm|swe|che|syr|twn|tjk|tza|tha|tls|tgo|tkl|ton|tto|tun|tur|tkm|tca|tuv|uga|ukr|are|gbr|umi|usa|ury|uzb|vut|ven|vnm|vgb|vir|wlf|esh|yem|zmb|zwe|ala
        
    """
    url = f"https://flixster.p.rapidapi.com/movies/get-upcoming"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if countryid:
        querystring['countryId'] = countryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, longitude: int=None, latitude: int=None, radius: int=50, zipcode: str='90002', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for movies, actors, theaters by term and phrase"
    longitude: The GEO longitude
        latitude: The GEO latitude
        radius: The radius
        zipcode: The postal code
        
    """
    url = f"https://flixster.p.rapidapi.com/search"
    querystring = {'query': query, }
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    if radius:
        querystring['radius'] = radius
    if zipcode:
        querystring['zipCode'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies_get_dvds(typemovie: str='NEW_RELEASE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get movies that watch at home"
    typemovie: One of the following : NEW&#95;RELEASE|COMING&#95;SOON
        
    """
    url = f"https://flixster.p.rapidapi.com/movies/get-dvds"
    querystring = {}
    if typemovie:
        querystring['typeMovie'] = typemovie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies_get_popularity(zipcode: str='90002', radius: int=50, longitude: int=None, latitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get popular movies"
    zipcode: The postal code
        radius: The radius
        longitude: The GEO longitude
        latitude: The GEO latitude
        
    """
    url = f"https://flixster.p.rapidapi.com/movies/get-popularity"
    querystring = {}
    if zipcode:
        querystring['zipCode'] = zipcode
    if radius:
        querystring['radius'] = radius
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies_get_opening(countryid: str='usa', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get opening movies"
    countryid: One of the following : afg|alb|dza|asm|and|ago|aia|ata|atg|arg|arm|abw|aus|aut|aze|bhs|bhr|bgd|brb|blr|bel|blz|ben|bmu|btn|bol|bes|bih|bwa|bvt|bra|iot|brn|bgr|bfa|bdi|cpv|khm|cmr|can|cym|caf|tcd|chl|chn|cxr|cck|col|com|cod|cog|cok|cri|hrv|cub|cuw|cyp|cze|civ|dnk|dji|dma|dom|ecu|egy|slv|gnq|eri|est|swz|eth|flk|fro|fji|fin|fra|guf|pyf|atf|gab|gmb|geo|deu|gha|gib|grc|grl|grd|glp|gum|gtm|ggy|gin|gnb|guy|hti|hmd|vat|hnd|hkg|hun|isl|ind|idn|irn|irq|irl|imn|isr|ita|jam|jpn|jey|jor|kaz|ken|kir|prk|kor|kwt|kgz|lao|lva|lbn|lso|lbr|lby|lie|ltu|lux|mac|mdg|mwi|mys|mdv|mli|mlt|mhl|mtq|mrt|mus|myt|mex|fsm|mda|mco|mng|mne|msr|mar|moz|mmr|nam|nru|npl|nld|ncl|nzl|nic|ner|nga|niu|nfk|mnp|nor|omn|pak|plw|pse|pan|png|pry|per|phl|pcn|pol|prt|pri|qat|mkd|rou|rus|rwa|reu|blm|shn|kna|lca|maf|spm|vct|wsm|smr|stp|sau|sen|srb|syc|sle|sgp|sxm|svk|svn|slb|som|zaf|sgs|ssd|esp|lka|sdn|sur|sjm|swe|che|syr|twn|tjk|tza|tha|tls|tgo|tkl|ton|tto|tun|tur|tkm|tca|tuv|uga|ukr|are|gbr|umi|usa|ury|uzb|vut|ven|vnm|vgb|vir|wlf|esh|yem|zmb|zwe|ala
        
    """
    url = f"https://flixster.p.rapidapi.com/movies/get-opening"
    querystring = {}
    if countryid:
        querystring['countryId'] = countryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

