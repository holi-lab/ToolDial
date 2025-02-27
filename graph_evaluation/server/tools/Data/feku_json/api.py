import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbookbyid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get Specific Book by ID"
    
    """
    url = f"https://feku-json1.p.rapidapi.com/api/v1/books/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feku-json1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbooks(sortdir: str='asc', sortby: str='id', pagesize: int=10, pagenumber: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get list of all books"
    
    """
    url = f"https://feku-json1.p.rapidapi.com/api/v1/books"
    querystring = {}
    if sortdir:
        querystring['sortDir'] = sortdir
    if sortby:
        querystring['sortBy'] = sortby
    if pagesize:
        querystring['pageSize'] = pagesize
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feku-json1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproductbyid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Specific Product by ID"
    
    """
    url = f"https://feku-json1.p.rapidapi.com/api/v1/products/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feku-json1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproducts(sortdir: str='asc', pagenumber: int=0, sortby: str='id', pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To Get all Product list"
    
    """
    url = f"https://feku-json1.p.rapidapi.com/api/v1/products"
    querystring = {}
    if sortdir:
        querystring['sortDir'] = sortdir
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if sortby:
        querystring['sortBy'] = sortby
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feku-json1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserbyid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To to Specific User by ID"
    
    """
    url = f"https://feku-json1.p.rapidapi.com/api/v1/users/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feku-json1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users(sortdir: str='asc', pagesize: int=10, pagenumber: int=0, sortby: str='id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To Get list of all Users"
    
    """
    url = f"https://feku-json1.p.rapidapi.com/api/v1/users"
    querystring = {}
    if sortdir:
        querystring['sortDir'] = sortdir
    if pagesize:
        querystring['pageSize'] = pagesize
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feku-json1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

