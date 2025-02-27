import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_data_by_research_a_specific_part_of_attribute_in_the_tag_selector(url: str, attribute: str, search: str, tag: str, pages: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data by “research a specific part of attribute” in the tag selector"
    
    """
    url = f"https://scrapemaster.p.rapidapi.com/api/search"
    querystring = {'url': url, 'attribute': attribute, 'search': search, 'tag': tag, }
    if pages:
        querystring['pages'] = pages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapemaster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_data_by_research_a_specific_word_string_in_the_tag_s_text(search: str, tag: str, url: str, pages: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data by “research a specific word/string” in the tag’s text"
    
    """
    url = f"https://scrapemaster.p.rapidapi.com/api/search"
    querystring = {'search': search, 'tag': tag, 'url': url, }
    if pages:
        querystring['pages'] = pages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapemaster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_data_by_id(is_id: str, tag: str, url: str, pages: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all data from a specific tag and its id."
    
    """
    url = f"https://scrapemaster.p.rapidapi.com/api/id"
    querystring = {'id': is_id, 'tag': tag, 'url': url, }
    if pages:
        querystring['pages'] = pages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapemaster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_data_by_class(is_class: str, tag: str, url: str, pages: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all data from a specific tag and its class attribute."
    
    """
    url = f"https://scrapemaster.p.rapidapi.com/api/class"
    querystring = {'class': is_class, 'tag': tag, 'url': url, }
    if pages:
        querystring['pages'] = pages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapemaster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_data_by_tag(tag: str, url: str, pages: str='3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all data from a specific tag."
    
    """
    url = f"https://scrapemaster.p.rapidapi.com/api/tag/"
    querystring = {'tag': tag, 'url': url, }
    if pages:
        querystring['pages'] = pages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapemaster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_page_s_content(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all the page's content from the URL given by the user."
    
    """
    url = f"https://scrapemaster.p.rapidapi.com/api/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapemaster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

