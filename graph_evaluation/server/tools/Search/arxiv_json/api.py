import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_paper_details(paper_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a paper, including code, citations, and references."
    paper_id: Arxiv ID of the paper. Obtained via /papers endpoint.
        
    """
    url = f"https://arxiv-json.p.rapidapi.com/papers/{paper_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arxiv-json.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_papers_by_author_and_or_keywords(keywords: str='network', sort_by: str='relevance', sort_order: str='descending', start: int=0, authors: str='geoffrey hinton', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search papers based on authors (comma separated) and/or keywords (comma separated). You can use either authors or keywords or both for searching."
    keywords: Comma separated keyword list.
        sort_by: Can be one of relevance, lastUpdatedDate, and submittedDate.
        sort_order: Can be either ascending or descending.
        start: Defines the index of the first returned result, using 0-based indexing.
        authors: Comma separated author list. 
        max_results: The number of results returned by the query, between 1 and 100.
        
    """
    url = f"https://arxiv-json.p.rapidapi.com/papers"
    querystring = {}
    if keywords:
        querystring['keywords'] = keywords
    if sort_by:
        querystring['sort_by'] = sort_by
    if sort_order:
        querystring['sort_order'] = sort_order
    if start:
        querystring['start'] = start
    if authors:
        querystring['authors'] = authors
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arxiv-json.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

