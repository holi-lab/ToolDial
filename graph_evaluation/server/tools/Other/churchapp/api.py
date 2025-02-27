import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_contacts(q: str=None, name: str=None, p: int=None, per_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a contact by name"
    q: Search by Name, Address, Job, Email, Telephone or Mobile
        name: Search by name (first, last, or both with "+" separating them, i.e. "first+last")
        p: If using pagination, define which page of contacts to return
        per_page: Define how many contacts should be returned per page
        
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/contacts"
    querystring = {}
    if q:
        querystring['q'] = q
    if name:
        querystring['name'] = name
    if p:
        querystring['p'] = p
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_contact(contact_id: int, tags: bool=None, keydates: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a contact by their unique ID"
    tags: If "true", this will return data for a specific contact, including any tags for the contact
        keydates: will return data for a specific contact, including any key dates for the contact
        
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/contact/{contact_id}"
    querystring = {}
    if tags:
        querystring['tags'] = tags
    if keydates:
        querystring['keydates'] = keydates
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_contacts_by_tag(tag_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return data for all contacts with the tag"
    
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/tag/{tag_id}/contacts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_dates_by_contact(contact_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/contact/{contact_id}/keydates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tag(tag_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return data for a specific tag with the defined ID"
    
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/tag/{tag_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tag_by_name(tag_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return data for a specific tag"
    
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/tag/{tag_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tags_by_contact(contact_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/contact/{contact_id}/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_contacts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all contacts, listed alphabetically"
    
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/contacts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return tags ordered alphabetically"
    
    """
    url = f"https://nicholasjohn-churchapp-v1.p.rapidapi.com/addressbook/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicholasjohn-churchapp-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

