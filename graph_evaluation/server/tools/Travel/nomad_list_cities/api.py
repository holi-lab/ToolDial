import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latin_america(sort_by: str='overall_score', page: str='1', sort: str='desc', size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Latin America cities sorted by overall score by default.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list/latin-america"
    querystring = {}
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def north_america(sort_by: str='overall_score', sort: str='desc', size: str='20', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get North America cities sorted by overall score by default.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list/north-america"
    querystring = {}
    if sort_by:
        querystring['sort_by'] = sort_by
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def middle_east(page: str='1', sort: str='desc', sort_by: str='overall_score', size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Middle East cities sorted by overall score by default.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list/middle-east"
    querystring = {}
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if sort_by:
        querystring['sort_by'] = sort_by
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def oceania(sort_by: str='overall_score', page: str='1', sort: str='desc', size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Oceania cities sorted by overall score by default.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list/oceania"
    querystring = {}
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def africa(sort: str='desc', sort_by: str='overall_score', page: str='1', size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get African cities sorted by overall score by default.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list/africa"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asia(sort: str='desc', sort_by: str='overall_score', size: str='20', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Asian cities sorted by overall score by default.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list/asia"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if sort_by:
        querystring['sort_by'] = sort_by
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def europe(sort: str='desc', page: str='1', sort_by: str='overall_score', size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get European cities sorted by overall score by default.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list/europe"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_citites(sort: str='desc', sort_by: str='overall_score', size: str='20', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cities sorted by overall score.
		
		Pagination: size & page 
		sort_by: you can sort by any numeric value like fore example internet_speed, temperatureC, cost_for_family_in_usd and so on.
		sort: asc | desc"
    
    """
    url = f"https://nomad-list-cities.p.rapidapi.com/nomad-list"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if sort_by:
        querystring['sort_by'] = sort_by
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nomad-list-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

