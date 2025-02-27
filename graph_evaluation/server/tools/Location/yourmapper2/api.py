import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dataset_list(maxlat: int, maxlon: int, minlat: int, minlon: int, format: str='json', sort: str='location', num: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of our data sets available within a Latitude/Longitude bounding box."
    maxlat: Maximum Latitude of bounding box.
        maxlon: Maximum Longitude of bounding box.
        minlat: Minimum Latitude of bounding box.
        minlon: Minimum Longitude of bounding box.
        format: Desired output format: html, xml, csv, json, rss
        sort: Sort of results: name, updated, rating, views, type, location
        num: Number of data sets to return
        
    """
    url = f"https://yourmapper2.p.rapidapi.com/datasetlist"
    querystring = {'maxlat': maxlat, 'maxlon': maxlon, 'minlat': minlat, 'minlon': minlon, }
    if format:
        querystring['format'] = format
    if sort:
        querystring['sort'] = sort
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yourmapper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def map_info(is_id: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detailed info about one data set."
    id: Unique identifier of the data set from 'datasetlist'
        format: json, xml, csv, html
        
    """
    url = f"https://yourmapper2.p.rapidapi.com/mapinfo"
    querystring = {'id': is_id, 'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yourmapper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def map_categories(is_id: int, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Category list and details associated with a data set."
    id: Unique identifier of the data set
        format: json, xml, csv, html, html2, html3
        
    """
    url = f"https://yourmapper2.p.rapidapi.com/mapcategories"
    querystring = {'id': is_id, 'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yourmapper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def marker_info(is_id: str, set: int, format: str, dist: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full details about a specific marker/location/point of data"
    id: Unique identifier of one data point
        set: Unique short ID of a dataset
        format: json, xml, csv, html, popup. Popup returns all the html code you need for a map popup bubble.
        dist: distance (in miles) of this point to the center of your search. Only needed for html and popup formats.
        
    """
    url = f"https://yourmapper2.p.rapidapi.com/markerinfo"
    querystring = {'id': is_id, 'set': set, 'format': format, }
    if dist:
        querystring['dist'] = dist
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yourmapper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markers(is_id: int, lat: int, lon: int, f: str, c: str='1,2,3', search: str='smith', start: str='2012-01-01', end: str='2012-12-31', num: int=5, center: int=1, radius: int=1, days: int=30, compact: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets closest marker/location data list around a center Latitude, Longitude for a certain data set.  This is the most important YourMapper API call."
    id: Short data set ID
        lat: Latitude of center point of search.
        lon: Longitude of center point of search
        f: Format you want the results as: json, xml, csv, html, rss, kml
        c: Comma separated list of category ids (from mapcategories) to include. Empty = all. 1 = ID 1.  1,2 = IDs 1 and 2.
        search: keyword to limit results. searches name, description fields.
        start: Start date if available. ISO yyyy-mm-dd format.
        end: End date if available. ISO yyyy-mm-dd format.
        num: Number of markers to return.  Default is 100. Can be 1 to 500.
        center: Include the informative center marker that shows center point and dataset data? Default is 1.
        radius: For the kml format only. Return a circle polyline to show search area. Default is 1.
        days: Instead of date range, you can specify number of days into the past to limit results.
        compact: Speeds up your query. If 1, less fields are returned since they can be gotten with markerinfo: description, categoryid, catcolor, geoid. Default is 0.
        
    """
    url = f"https://yourmapper2.p.rapidapi.com/markers"
    querystring = {'id': is_id, 'lat': lat, 'lon': lon, 'f': f, }
    if c:
        querystring['c'] = c
    if search:
        querystring['search'] = search
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    if num:
        querystring['num'] = num
    if center:
        querystring['center'] = center
    if radius:
        querystring['radius'] = radius
    if days:
        querystring['days'] = days
    if compact:
        querystring['compact'] = compact
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yourmapper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

