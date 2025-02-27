import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest(latest: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "latest manga"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/"
    querystring = {}
    if latest:
        querystring['latest'] = latest
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_page7(top: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page7"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/top/7"
    querystring = {}
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_page6(top: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page6"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/top/6"
    querystring = {}
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_page5(top: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page5"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/top/5"
    querystring = {}
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_page4(top: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page4"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/top/4"
    querystring = {}
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_page3(top: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page3"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/top/3"
    querystring = {}
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_page2(top: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page2"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/top/2"
    querystring = {}
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_page1(top: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page1"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/top"
    querystring = {}
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page7(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page7"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/7"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page6(get_6: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page6"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/6"
    querystring = {}
    if get_6:
        querystring['6'] = get_6
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page5(get_5: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page5"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/5"
    querystring = {}
    if get_5:
        querystring['5'] = get_5
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page4(get_4: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page4"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/4"
    querystring = {}
    if get_4:
        querystring['4'] = get_4
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page3(get_3: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page3"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/3"
    querystring = {}
    if get_3:
        querystring['3'] = get_3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page2(get_2: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "page2"
    
    """
    url = f"https://mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com/2"
    querystring = {}
    if get_2:
        querystring['2'] = get_2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangadb-with-lots-of-top-rated-manga-manwha-manuha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

