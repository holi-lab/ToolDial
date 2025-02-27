import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def articlesearch_json(end_date: str=None, q: str=None, fq: str=None, facet_fields: str='day_of_week', page: int=None, facet: str=None, sort: str='newest', fl: str=None, facet_filter: str=None, begin_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for NYT articles by keywords, filters and facets."
    end_date: End date (e.g. 20121231)
        q: Query
        fq: Filter query
        facet_fields: Facets
        page: Page number (0, 1, ...)
        facet: Whether to show facet counts
        sort: Sort order
        fl: Field list
        facet_filter: Have facet counts use filters
        begin_date: Begin date (e.g. 20120101)
        
    """
    url = f"https://ny-times-article-search.p.rapidapi.com/articlesearch.json"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if q:
        querystring['q'] = q
    if fq:
        querystring['fq'] = fq
    if facet_fields:
        querystring['facet_fields'] = facet_fields
    if page:
        querystring['page'] = page
    if facet:
        querystring['facet'] = facet
    if sort:
        querystring['sort'] = sort
    if fl:
        querystring['fl'] = fl
    if facet_filter:
        querystring['facet_filter'] = facet_filter
    if begin_date:
        querystring['begin_date'] = begin_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-article-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

