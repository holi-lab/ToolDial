import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reviews_search_json(publication_date: str=None, query: str=None, reviewer: str=None, offset: int=None, opening_date: str=None, critics_pick: str='Y', order: str='by-opening-date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for movie reviews.  Supports filtering by Critics' Pick.
		"
    publication_date: Review publication date range. Start and end dates separated by colon (e.g. 1930-01-01:1940-01-01).
        query: Search keyword (e.g. lebowski).
        reviewer: Filter by reviewer byline (e.g. Stephen Holden).
        offset: Sets the starting point of the result set (0, 20, ...).  Used to paginate thru results. Defaults to 0. The has_more field indicates the response has more results.
        opening_date: U.S. opening date range. Start and end dates separated by colon (e.g. 1930-01-01:1940-01-01).
        critics_pick: Set to Y to only show critics' picks.  Otherwise shows both.
        order: Field to order results by (e.g. by-publication-date).
        
    """
    url = f"https://ny-times-movie-reviews.p.rapidapi.com/reviews/search.json"
    querystring = {}
    if publication_date:
        querystring['publication-date'] = publication_date
    if query:
        querystring['query'] = query
    if reviewer:
        querystring['reviewer'] = reviewer
    if offset:
        querystring['offset'] = offset
    if opening_date:
        querystring['opening-date'] = opening_date
    if critics_pick:
        querystring['critics-pick'] = critics_pick
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-movie-reviews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def critics_reviewer_json(reviewer: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get movie critics. You can either specify the reviewer name or use "all", "full-time", or "part-time".
		"
    reviewer: Reviewer name or "all" for all reviewers, "full-time" for full-time reviewers, or "part-time" for part-time reviewers.
        
    """
    url = f"https://ny-times-movie-reviews.p.rapidapi.com/critics/{reviewer}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-movie-reviews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_type_json(type: str, offset: int=0, order: str='by-opening-date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get movie reviews.  Can filter to only return Critics' Picks.
		Supports ordering results by-publication-date or by-opening-date.
		Use offset to paginate thru results, 20 at a time.
		"
    type: Filter by critics' pick or not.
        offset: Sets the starting point of the result set.  Needs to be multiple of 20.
        order: How to order the results.
        
    """
    url = f"https://ny-times-movie-reviews.p.rapidapi.com/reviews/{type}.json"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-movie-reviews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

