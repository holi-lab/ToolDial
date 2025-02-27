import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def neighbors(is_id: str, offset: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Neighbors - Return a list of neighbors sorted by relative rank."
    offset: Skip first {offset} neighbors for context related to parent {_id}
        limit: Max Number of neighbors related to parent {_id}
        
    """
    url = f"https://_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com/neighbors/{id}/"
    querystring = {'_id': is_id, 'offset': offset, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_base_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Welcoming message with contact details to developer."
    
    """
    url = f"https://_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search (fuzzy) function. Based on Wikimedia API, returned topics are mapped to entities in factual knowledge graph."
    q: Keywords to query. Keywords identify topics' name.
        
    """
    url = f"https://_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com/fuzzyautocomplete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def context_between_multiple_entities(is_id: str, second: str='PAyD8rYKqxEM47Vk', n_th: str='59v4aQmEBlkayeGZ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return graph of connections between a set of topics. Max number of topics is 5."
    id: ID of first entity to query, return a sub-graph of entities IDs in path URL
        second: ID of a second entity to query
        n_th: ID of a n-th entity to query. Response is restricted to max 5 entities' IDs in path URL
        
    """
    url = f"https://_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com/insight/{is_id}/{second}/{n_th}/"
    querystring = {}
    if second:
        querystring['_second'] = second
    if n_th:
        querystring['_n-th'] = n_th
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def context_of_a_topic(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return insights about a topic. Insight is a sub-graph of first neighbours related to a parent {_id}."
    id: Return the context of an entity as a subgraph of 6 neighbours for each entity's neighbour
        
    """
    url = f"https://_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com/insight/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "_next-factual-knowledge-graph-wikipedia-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

