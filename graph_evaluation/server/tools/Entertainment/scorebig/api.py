import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://scorebig-scorebig.p.rapidapi.com/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scorebig-scorebig.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def performers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://scorebig-scorebig.p.rapidapi.com/performers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scorebig-scorebig.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://scorebig-scorebig.p.rapidapi.com/events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scorebig-scorebig.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_search(event_id: str='3D4E6F14-D4DA-4B92-9DC3-410A6EEBB43D', query: str='los angeles lakers', name: str='Cirque%20du%20Soleil', performer_name: str='Rams', performer_id: str='BAB8EB10-39C9-49D9-9563-78288B2DC69A, 4104E3BA-90EB-4BA0-8EE1-7DED6B4928BF', secondary_performer_id: str='BAB8EB10-39C9-49D9-9563-78288B2DC69A, 4104E3BA-90EB-4BA0-8EE1-7DED6B4928BF', venue: str='staples center', city: str='Boston', metro_region_id: str='LAX', category: str='sports', category_id: str='4EA5447D-DECD-4C0B-A0C4-0EBB80495DC5, AD98B56A-1F5A-401D-8DBD-FDBF370200ED', start_date: str='09-01-2012-', end_date: str='09-30-2012-', deal_rating: str='1,2', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    event_id: A single event id.
        query: Free form text for a general search (event, city, venue, performer, etc)
        name: Free form event name
        performer_name: Free form performer name
        performer_id: IDs of the performers to restrict the search to (home games only)
        secondary_performer_id: IDs of the performers to restrict the search to (away games only)
        venue: Free form venue name
        city: Free form city name
        metro_region_id: IDs of the metro regions to restrict the search to.  Multiple metro region IDs should be separated by commas.
        category: Free form metro category name
        category_id: IDs of the category to restrict the search to.  Multiple category ids should be separated by commas.
        start_date: Only include events after this date and time
        end_date: Only include events up to this date and time
        deal_rating: Restrict search to certain deal ratings
        max_results: Maximum number of results to return.  Default is 200
        
    """
    url = f"https://scorebig-scorebig.p.rapidapi.com/Events"
    querystring = {}
    if event_id:
        querystring['event_id'] = event_id
    if query:
        querystring['query'] = query
    if name:
        querystring['name'] = name
    if performer_name:
        querystring['performer_name'] = performer_name
    if performer_id:
        querystring['performer_id'] = performer_id
    if secondary_performer_id:
        querystring['secondary_performer_id'] = secondary_performer_id
    if venue:
        querystring['venue'] = venue
    if city:
        querystring['city'] = city
    if metro_region_id:
        querystring['metro_region_id'] = metro_region_id
    if category:
        querystring['category'] = category
    if category_id:
        querystring['category_id'] = category_id
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if deal_rating:
        querystring['deal_rating'] = deal_rating
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scorebig-scorebig.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of valid categories."
    
    """
    url = f"https://scorebig-scorebig.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scorebig-scorebig.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

