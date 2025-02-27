import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(v: str, format: str, callback: str, q: str, l: str, sort: str, radius: str, st: str, jt: str, start: str, limit: str, fromage: str, highlight: str, filter: str, latlong: str, co: str, chnl: str, userip: str, useragent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    v: Version. Which version of the API you wish to use. All publishers should be using version 2. Currently available versions are 1 and 2. This parameter is required.
        format: Format. Which output format of the API you wish to use. The options are "xml" and "json." If omitted or invalid, the XML format is used.
        callback: Callback. The name of a javascript function to use as a callback to which the results of the search are passed. This only applies when format=json. For security reasons, the callback name is restricted letters, numbers, and the underscore character.
        q: Callback. The name of a javascript function to use as a callback to which the results of the search are passed. This only applies when format=json. For security reasons, the callback name is restricted letters, numbers, and the underscore character.
        l: Location. Use a postal code or a "city, state/province/region" combination.
        sort: Sort by relevance or date. Default is relevance.
        radius: Distance from search location ("as the crow flies"). Default is 25.
        st: Site type. To show only jobs from job boards use 'jobsite'. For jobs from direct employer websites use 'employer'.
        jt: Job type. Allowed values: "fulltime", "parttime", "contract", "internship", "temporary".
        start: Start results at this result number, beginning with 0. Default is 0.
        limit: Maximum number of results returned per query. Default is 10
        fromage: Number of days back to search.
        highlight: Setting this value to 1 will bold terms in the snippet that are also present in q. Default is 0.
        filter: Filter duplicate results. 0 turns off duplicate job filtering. Default is 1.
        latlong: If latlong=1, returns latitude and longitude information for each job result. Default is 0.
        co: Search within country specified. Default is us. See below for a complete list of supported countries.
        chnl: Channel Name: Group API requests to a specific channel
        userip: The IP number of the end-user to whom the job results will be displayed. This field is required.
        useragent: The User-Agent (browser) of the end-user to whom the job results will be displayed. This can be obtained from the "User-Agent" HTTP request header from the end-user. This field is required.
        
    """
    url = f"https://indeed-indeed.p.rapidapi.com/apisearch"
    querystring = {'v': v, 'format': format, 'callback': callback, 'q': q, 'l': l, 'sort': sort, 'radius': radius, 'st': st, 'jt': jt, 'start': start, 'limit': limit, 'fromage': fromage, 'highlight': highlight, 'filter': filter, 'latlong': latlong, 'co': co, 'chnl': chnl, 'userip': userip, 'useragent': useragent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indeed-indeed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

