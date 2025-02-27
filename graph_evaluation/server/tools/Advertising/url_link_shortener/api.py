import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_a_new_link(destination: str, domain_id: str=None, slashtag: str=None, domain_fullname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a new link"
    destination: The destination URL you want your branded short link to point to
        domain_id: The unique id of the branded domain. If not specified, rebrand.ly is used
        slashtag: The keyword portion of your branded short link
        domain_fullname: The unique name of the branded domain, to be used in place of domain[id] in special cases. Precedence will be given to domain[id] value.
        
    """
    url = f"https://url-link-shortener.p.rapidapi.com/v1/links/new"
    querystring = {'destination': destination, }
    if domain_id:
        querystring['domain[id]'] = domain_id
    if slashtag:
        querystring['slashtag'] = slashtag
    if domain_fullname:
        querystring['domain[fullName]'] = domain_fullname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-link-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_domains(active: bool=None, orderby: str='createdAt', last: str=None, limit: int=100, orderdir: str='desc', type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of domains"
    active: Filter branded domains depending on whether they can be used to brand short links or not
        orderby: Sorting criteria to apply to your branded domains collection among createdAt, updatedAt and fullName.
        last: The id of the last domain you fetched, see Infinite Scrolling section
        limit: How many branded domains to load
        orderdir: Sorting direction to apply to your branded short links collection among desc and asc.
        type: Filter branded domains depending on their type (owned by user, user, or service domains like rebrand.ly, service)
        
    """
    url = f"https://url-link-shortener.p.rapidapi.com/v1/domains"
    querystring = {}
    if active:
        querystring['active'] = active
    if orderby:
        querystring['orderBy'] = orderby
    if last:
        querystring['last'] = last
    if limit:
        querystring['limit'] = limit
    if orderdir:
        querystring['orderDir'] = orderdir
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-link-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

