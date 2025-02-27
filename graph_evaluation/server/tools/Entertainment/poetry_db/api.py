import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def title(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all poem titles on Poetry DB"
    
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/title"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_title(title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems on Poetry DB with a specific title"
    title: The title of the poem, or part of the title of the poem.
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/title/{title}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linecount_linecount(linecount: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems on Poetry DB with a specific number of lines"
    linecount: The number of lines in a poem.
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/linecount/{linecount}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def author(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all poets on Poetry DB"
    
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/author"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def author_title_author_title(author: str, title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems with a specific title by a specific author"
    author: The poet, or part of the poet's name
        title: The title of the poem, or part of the title of the poem.
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/author,title/{author};{title}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def author_author_abs(author: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems by a specific author"
    author: The exact name of the poet
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/author/{author}:abs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_title_abs(title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems with a specific title"
    title: The exact title of the poem
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/title/{title}:abs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lines_line(line: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems on Poetry DB containing a specific line"
    line: The line of a poem, or part of the line of a poem.
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/lines/{line}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def author_author(author: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems on Poetry DB by a specific poet"
    author: The poet, or part of the poet's name.
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/author/{author}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lines_line_abs(line: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All poems that contain a specific line"
    line: The exact line in the poem
        
    """
    url = f"https://thundercomb-poetry-db-v1.p.rapidapi.com/lines/{line}:abs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thundercomb-poetry-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

