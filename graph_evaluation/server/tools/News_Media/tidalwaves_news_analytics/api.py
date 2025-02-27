import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_the_latest_articles(withkeywords: str='True', withlocations: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all articles from the latest batch."
    withkeywords: Include related keywords for each article.
        withlocations: Include related locations for each article.
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/articles/latest"
    querystring = {}
    if withkeywords:
        querystring['withKeywords'] = withkeywords
    if withlocations:
        querystring['withLocations'] = withlocations
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_the_latest_stories(witharticles: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all stories from the latest batch."
    witharticles: Include related articles (with keywords & locations) for each story.
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/stories/latest"
    querystring = {}
    if witharticles:
        querystring['withArticles'] = witharticles
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_article_locations(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quick location fetch for specific article"
    
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/articles/{is_id}/locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_article_keywords(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quick keyword fetch for specific article"
    
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/articles/{is_id}/keywords"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_articles(orderdir: str='asc', orderby: str='title', is_from: str='2020-10-17T11:00', maxtone: int=5, offset: int=100, limit: int=10, to: str='2020-10-17T12:00', mintone: int=-1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query flat article data."
    orderdir: Row ordering direction. Either "asc" or "desc".
        orderby: Row ordering. Available columns: "title", "tone", and "timestamp".
        is_from: Min. publishing timestamp.
        maxtone: Max. tone sentiment.
        offset: Number of rows to offset by.
        limit: Number of rows to return.
        to: Max. publishing timestamp.
        mintone: Min. tone sentiment.
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/articles"
    querystring = {}
    if orderdir:
        querystring['orderDir'] = orderdir
    if orderby:
        querystring['orderBy'] = orderby
    if is_from:
        querystring['from'] = is_from
    if maxtone:
        querystring['maxTone'] = maxtone
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if to:
        querystring['to'] = to
    if mintone:
        querystring['minTone'] = mintone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_stories(to: str='2020-10-17T12:00', mintone: int=-1, orderby: str='gist', offset: int=100, limit: int=10, maxtone: int=5, orderdir: str='asc', is_from: str='2020-10-17T11:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query trending stories with their related articles."
    to: Max. publishing timestamp.
        mintone: Min. average tone sentiment.
        orderby: Row ordering. Available columns: "gist", "avg_tone", and "timestamp".
        offset: Number of rows to offset by.
        limit: Number of rows to return.
        maxtone: Max. average tone sentiment.
        orderdir: Row ordering direction. Either "asc" or "desc".
        is_from: Min. publishing timestamp.
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/stories"
    querystring = {}
    if to:
        querystring['to'] = to
    if mintone:
        querystring['minTone'] = mintone
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if maxtone:
        querystring['maxTone'] = maxtone
    if orderdir:
        querystring['orderDir'] = orderdir
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_keywords(orderby: str='label', offset: int=100, exact: str='True', limit: int=10, orderdir: str='asc', query: str='brexit', type: str='misc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query keywords according to search string and other parameters."
    orderby: Row ordering. Available columns: "id", "label".
        offset: Number of rows to offset by.
        exact: Flag to match for exact query string, or allow partial matching.
        limit: Number of rows to return.
        orderdir: Row ordering direction. Either "asc" or "desc".
        query: Search query string.
        type: Category filter. Available categories: "themes", "people", "orgs", and "misc".
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/keywords"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    if exact:
        querystring['exact'] = exact
    if limit:
        querystring['limit'] = limit
    if orderdir:
        querystring['orderDir'] = orderdir
    if query:
        querystring['query'] = query
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_keyword_articles(is_id: str, maxtone: int=5, mintone: int=-1, orderdir: str='asc', to: str='2020-10-17T12:00', limit: int=10, is_from: str='2020-10-17T11:00', orderby: str='title', offset: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query keywords according to search string and other parameters."
    maxtone: Max. tone sentiment.
        mintone: Min. tone sentiment.
        orderdir: Row ordering direction. Either "asc" or "desc".
        to: Max. publishing timestamp.
        limit: Number of rows to return.
        is_from: Min. publishing timestamp.
        orderby: Row ordering. Available columns: "title", "tone", and "timestamp".
        offset: Number of rows to offset by.
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/keywords/{is_id}/articles"
    querystring = {}
    if maxtone:
        querystring['maxTone'] = maxtone
    if mintone:
        querystring['minTone'] = mintone
    if orderdir:
        querystring['orderDir'] = orderdir
    if to:
        querystring['to'] = to
    if limit:
        querystring['limit'] = limit
    if is_from:
        querystring['from'] = is_from
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_locations(limit: int=10, orderby: str='tag', query: str='eng', offset: int=100, country: str='GB', exact: str='True', orderdir: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query keywords according to search string and other parameters."
    limit: Number of rows to return.
        orderby: Row ordering. Available columns: "country_code", "tag", "lat", "long".
        query: Search query string.
        offset: Number of rows to offset by.
        country: ISO 3306 Alpha-2 country code filter.
        exact: Flag to match for exact query string, or allow partial matching.
        orderdir: Row ordering direction. Either "asc" or "desc".
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/locations"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if orderby:
        querystring['orderBy'] = orderby
    if query:
        querystring['query'] = query
    if offset:
        querystring['offset'] = offset
    if country:
        querystring['country'] = country
    if exact:
        querystring['exact'] = exact
    if orderdir:
        querystring['orderDir'] = orderdir
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_location_articles(is_id: str, maxtone: int=5, to: str='2020-10-17T12:00', orderby: str='title', offset: int=100, limit: int=10, mintone: int=-1, orderdir: str='asc', is_from: str='2020-10-17T11:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query keywords according to search string and other parameters."
    maxtone: Max. tone sentiment.
        to: Max. publishing timestamp.
        orderby: Row ordering. Available columns: "title", "tone", and "timestamp".
        offset: Number of rows to offset by.
        limit: Number of rows to return.
        mintone: Min. tone sentiment.
        orderdir: Row ordering direction. Either "asc" or "desc".
        is_from: Min. publishing timestamp.
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/locations/{is_id}/articles"
    querystring = {}
    if maxtone:
        querystring['maxTone'] = maxtone
    if to:
        querystring['to'] = to
    if orderby:
        querystring['orderBy'] = orderby
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if mintone:
        querystring['minTone'] = mintone
    if orderdir:
        querystring['orderDir'] = orderdir
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_api_health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Performs various system checks, such as connection to the DB and query latency, and returns the result. Use this endpoint to confirm the API is available."
    
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_batches(offset: int=100, to: str='2020-10-17T12:00', is_from: str='2020-10-17T11:00', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View batch history."
    offset: Number of rows to offset by.
        to: Max. publishing timestamp.
        is_from: Min. publishing timestamp.
        limit: Number of rows to return.
        
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/batches"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_the_latest_batch(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Useful for polling the latest data."
    
    """
    url = f"https://tidalwaves-news-analytics2.p.rapidapi.com/batches/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tidalwaves-news-analytics2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

