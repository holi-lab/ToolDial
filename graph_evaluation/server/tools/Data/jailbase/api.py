import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recent(source_id: str, page: int=None, json_callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent arrests"
    source_id: The id of a specific organization to search (use 'az-mcso' for test).  Full list at http://www.jailbase.com/api/#sources_list
        page: The page number to return.  Only 10 records are returned per page.  See total_records, current_page and next_page values in the results.
        json_callback: If using JSONP, specify the function name here.
        
    """
    url = f"https://jailbase-jailbase.p.rapidapi.com/recent/"
    querystring = {'source_id': source_id, }
    if page:
        querystring['page'] = page
    if json_callback:
        querystring['json_callback'] = json_callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jailbase-jailbase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(source_id: str, last_name: str, first_name: str=None, page: int=None, json_callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for individuals by name"
    source_id: The id of a specific organization to search.  Full list at http://www.jailbase.com/api/#sources_list
        last_name: The last name to search for, partial names accepted
        first_name: The first name to search for, partial names accepted
        page: The page number to return. Only 10 records are returned per page. See total_records, current_page and next_page values in the results. Default: 1.
        json_callback: If using JSONP, specify the function name here
        
    """
    url = f"https://jailbase-jailbase.p.rapidapi.com/search/"
    querystring = {'source_id': source_id, 'last_name': last_name, }
    if first_name:
        querystring['first_name'] = first_name
    if page:
        querystring['page'] = page
    if json_callback:
        querystring['json_callback'] = json_callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jailbase-jailbase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sources(json_callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All the organizations we collect information for"
    json_callback: If using JSONP, specify the function name here.
        
    """
    url = f"https://jailbase-jailbase.p.rapidapi.com/sources/"
    querystring = {}
    if json_callback:
        querystring['json_callback'] = json_callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jailbase-jailbase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

