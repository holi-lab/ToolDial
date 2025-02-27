import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def webcams_list_nearby_lat_lng_radius(radius: int, lng: int, lat: int, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of webcams near the given coordinates {lat},{lng} up to the given {radius} in kilometers. Required: {lat}, {lng}, {radius}. The maximum value for {radius} is 250."
    radius: Maximum distance in kilometers.
        lng: WGS84 longitude.
        lat: WGS84 latitude.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/nearby={lat},{lng},{radius}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_category_category_category(category: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of webcams according to the listed categories. Multiple categories must be separated by comma. Required: at least one {category}."
    category: Comma separated list of category names.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/category={category}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_continent_continent_continent(continent: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of webcams according to the listed continents. Multiple continents must be separated by comma. Required: at least one {continent}."
    continent: Possible values for {continent} are: "AF" (Africa), "AN" (Antarctica), "AS" (Asia), "EU" (Europe), "NA" (North America), "OC" (Oceania), or "SA" (South America)
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/continent={continent}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_modifier_modifier(modifier: str, lang: str='en', show: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of webcams according to the applied {modifier}s. Initially, the list contains all webcams. The resulting list of webcams from applying a modifer will be used as input for the following modifier. modifiers are processed from left to right. With every applied modifier, the list of webcams will be narrowed down. Possible values for {modifier} are: "webcam" (initial list of webcams), "nearby" (list of webcams around a position), "bbox" (list of webcams in an area), "category" (list of webcams in a category), "continent" (list of webcams in a continent), "country" (list of webcams in a country), "region" (list of webcams in a region of a country), "exclude" (exclude webcams from a list of webcams), "orderby" (order the list of webcams), "limit" (slice the list of webcams). "webcam", if given, is always applied first. "exclude", "orderby", and "limit" are always applied (even if not explicitly given) in this order and always after "webcam", "nearby", "bbox", "category", "continent", "country", and "region" have been applied.  If none of "webcam", "nearby", "bbox", "category", "continent", "country", and "region" are applied, then all webcams are in the list before applying "exclude", "orderby", and "limit".  Please refer to the description of the various {modifier}s to learn more about their parameter and defaults.  A {modifier} may be applied only once. If a {modifier} is listed multiple times, it may only applied once."
    lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/{modifier}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_country_country_country(country: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of webcams according to the listed country. Multiple countries must be separated by comma. Required: at least one {country}. Possible values are ISO 3166-1-alpha-2 codes."
    country: Comma separated ISO 3166-1-alpha-2 codes for the countries.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/country={country}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_region_region_region(region: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of webcams according to the listed region. Multiple regions must be separated by comma. Required: at least one {region}. Possible values are ISO 3166-1-alpha-2 codes for the country and a region code separated by a dot."
    region: Comma separated list of ISO 3166-1-alpha-2 codes for the country and a region code separated by a dot.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/region={region}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_bbox_ne_lat_ne_lng_sw_lat_sw_lng(ne_lat: int, sw_lng: int, sw_lat: int, ne_lng: int, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of the webcams in the bounding box given by north-east ({ne_lat},{ne_lng}) and south-west ({sw_lat},{sw_lng}) coordinates. Required: {ne_lat},{ne_lng},{sw_lat},{sw_lng}."
    ne_lat: North-east WGS84 latitude of the bounding box.
        sw_lng: South-west WGS84 longitude of the bounding box.
        sw_lat: South-west WGS84 latitude of the bounding box.
        ne_lng: North-east WGS84 longitude of the bounding box.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/bbox={ne_lat},{ne_lng},{sw_lat},{sw_lng}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_exclude_webcamid_webcamid(webcamid: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Exclude the {webcamid} from a list. Multiple {webcamid} must be separated by comma. Required: at least one {webcamid}"
    webcamid: Comma separated list of webcamids to exclude from result.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/exclude={webcamid}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_orderby_order_sort(sort: str, order: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns the list of webcams ordered by {order}. The optional sorting direction is given by {sort}. Required {order}."
    sort: Possible values are: "asc" (ascending), or "desc" (descending).
        order: Possible values are: "popularity" (default order: "desc"), "hotness" (default order: "desc"), "new" (default order: "desc"), "recent" (default order: "desc"), "random" (default order: "asc"), or "distance" (default order: "asc", only available if the modifier "nearby" has been applied).
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/orderby={order},{sort}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_limit_limit_offset(limit: int, offset: int=0, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns the list of webcams sliced by {limit}. The optional offset is given by {offset}. Required: {limit}. The maximum value for {limit} is 50. {offset} defaults to 0. If limit is not given, then a default of limit=10 is applied."
    limit: Maximum number of webcams in the result list.
        offset: Offset for the first item in the result list.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/limit={limit},{offset}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_map_ne_lat_ne_lng_sw_lat_sw_lng_zoom(ne_lat: int, ne_lng: int, sw_lat: int, sw_lng: int, zoom: int, lang: str=None, show: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of webcams optimized for displaying on a map for a given bounding box and {zoom} level. The value for the zoom level is compatible with the Google Maps zoom level."
    ne_lat: North-east WGS84 latitude of the bounding box.
        ne_lng: North-east WGS84 longitude of the bounding box.
        sw_lat: South-west WGS84 latitude of the bounding box.
        sw_lng: South-west WGS84 longitude of the bounding box.
        zoom: A zoom level compatible with Google Maps.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/map/{ne_lat},{ne_lng},{sw_lat},{sw_lng},{zoom}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_property_property_property(property: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of webcams according to the listed {property}. Multiple {property} must be separated by comma. Required: at least one {property}. Possible values are "editors" (featured from the editors), "hd" (high resolution webcams), and "live" (webcams with a live stream)."
    lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/property={property}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webcams_list_webcam_webcamid_webcamid(webcamid: str, lang: str='en', show: str='webcams:image,location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a modifier. Returns a list of webcams according to the listed {webcamid}s. Multiple {webcamid}s must be separated by comma. Required: at least one {webcamid}."
    webcamid: Comma separated list of webcamids that are in the initial list. If this modifier ist not applied, all available webcams will be in the initial list.
        lang: Localize the results to this language, if available. Default: "en".
        show: Content to be listed in the response. Possible values are: "webcams", "categories", "continents", "countries", "regions", "properties". Default is "webcams".
        
    """
    url = f"https://webcamstravel.p.rapidapi.com/webcams/list/webcam={webcamid}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if show:
        querystring['show'] = show
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webcamstravel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

