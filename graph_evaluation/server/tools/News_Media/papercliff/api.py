import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def combinations(to: str=None, terms: str=None, offset: int=None, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the 100 most popular combinations/triples of keywords with the corresponding number of news agencies and articles from which they have been cited"
    to: Narrows down the results to articles published before the provided date-time in UTC. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T15:30`. Date-times of the future do not affect the result.
        terms: Narrows down the results to articles that contain all the provided keywords. The terms should consist of one to three words separated by a dash.  Example value: `election-campaign`.
        offset: Omits a number of combinations
        is_from: Narrows down the results to articles published after the provided date-time in UTC. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T13:45`. Date-times older than a week do not affect the result.
        
    """
    url = f"https://papercliff.p.rapidapi.com/combinations"
    querystring = {}
    if to:
        querystring['to'] = to
    if terms:
        querystring['terms'] = terms
    if offset:
        querystring['offset'] = offset
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "papercliff.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeline(is_from: str=None, terms: str=None, offset: int=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the 100 most recent articles and their keywords"
    is_from: Narrows down the results to articles published after the provided date-time in UTC. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T13:45`. Date-times older than a week do not affect the result.
        terms: Narrows down the results to articles that contain all the provided keywords. The terms should consist of one to three words separated by a dash.  Example value: `election-campaign`.
        offset: Omits a number of articles
        to: Narrows down the results to articles published before the provided date-time in UTC. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T15:30`. Date-times of the future do not affect the result.
        
    """
    url = f"https://papercliff.p.rapidapi.com/timeline"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if terms:
        querystring['terms'] = terms
    if offset:
        querystring['offset'] = offset
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "papercliff.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def overview(is_from: str=None, to: str=None, terms: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns summary statistics about how many keywords have been found and how many articles and agencies papercliff looked at"
    is_from: Narrows down the results to articles published after the provided date-time. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T13:45`. Date-times older than a week do not affect the result.
        to: Narrows down the results to articles published before the provided date-time. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T15:30`. Date-times of the future do not affect the result.
        terms: Narrows down the results to articles that contain all the provided keywords. The terms should consist of one to three words separated by a dash.  Example value: `election-campaign`.
        
    """
    url = f"https://papercliff.p.rapidapi.com/overview"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if terms:
        querystring['terms'] = terms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "papercliff.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history(terms: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the number of articles published daily during the last week and the number of corresponding news agencies that created those articles"
    terms: Narrows down the results to articles that contain all the provided keywords. The terms should consist of one to three words separated by a dash.  Example value: `election-campaign`.
        
    """
    url = f"https://papercliff.p.rapidapi.com/history"
    querystring = {}
    if terms:
        querystring['terms'] = terms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "papercliff.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keywords(terms: str=None, offset: int=None, is_from: str=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the 100 most popular keywords with the corresponding number of news agencies and articles from which they have been cited"
    terms: Narrows down the results to articles that contain all the provided keywords. The terms should consist of one to three words separated by a dash.  Example value: `election-campaign`.
        offset: Omits a number of keywords
        is_from: Narrows down the results to articles published after the provided date-time in UTC. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T13:45`. Date-times older than a week do not affect the result.
        to: Narrows down the results to articles published before the provided date-time in UTC. The format should be `yyyy-MM-dd'T'HH:mm`. Example value: `2022-09-18T15:30`. Date-times of the future do not affect the result.
        
    """
    url = f"https://papercliff.p.rapidapi.com/keywords"
    querystring = {}
    if terms:
        querystring['terms'] = terms
    if offset:
        querystring['offset'] = offset
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "papercliff.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

