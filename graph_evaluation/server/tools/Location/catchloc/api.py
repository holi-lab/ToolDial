import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def checking_the_information_of_the_registered_object_api(api_key: str, timestamp: str, cert_key: str, page_size: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for checking for registered objects and last location information"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.get.member.info.all.php"
    querystring = {'api_key': api_key, 'timestamp': timestamp, 'cert_key': cert_key, }
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checking_the_recent_location_information_api(cert_key: str, timestamp: str, member_key: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for checking the last collected location of an object."
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.get.member.location.last.php"
    querystring = {'cert_key': cert_key, 'timestamp': timestamp, 'member_key': member_key, 'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checking_section_definition_location_data_api(member_key: str, cert_key: str, timestamp: str, to_date: str, from_date: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for checking targeted section of location for specified time of an object."
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.get.member.location.list.php"
    querystring = {'member_key': member_key, 'cert_key': cert_key, 'timestamp': timestamp, 'to_date': to_date, 'from_date': from_date, 'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_for_registered_group_list(api_key: str=None, cert_key: str=None, timestamp: str=None, api: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access for group information.
		
		required parameter : api (api.common.group.get.list)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    if cert_key:
        querystring['cert_key'] = cert_key
    if timestamp:
        querystring['timestamp'] = timestamp
    if api:
        querystring['api'] = api
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_for_device_s_group_list(cert_key: str=None, member_key: str=None, api: str=None, timestamp: str=None, api_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access for location object's group list.
		
		required parameter : api (api.common.group.get.object.group.list)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {}
    if cert_key:
        querystring['cert_key'] = cert_key
    if member_key:
        querystring['member_key'] = member_key
    if api:
        querystring['api'] = api
    if timestamp:
        querystring['timestamp'] = timestamp
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_for_group_s_device_list(group_key: str=None, timestamp: str=None, api_key: str=None, api: str=None, cert_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access for location object group's device list.
		
		required parameter : api(api.common.group.get.group.object.list)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {}
    if group_key:
        querystring['group_key'] = group_key
    if timestamp:
        querystring['timestamp'] = timestamp
    if api_key:
        querystring['api_key'] = api_key
    if api:
        querystring['api'] = api
    if cert_key:
        querystring['cert_key'] = cert_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_for_creating_group(cert_key: str=None, api_key: str=None, group_name: str=None, timestamp: str=None, api: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access for location object's group designation and creation.
		
		required parameter : api (api.common.group.set.add)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {}
    if cert_key:
        querystring['cert_key'] = cert_key
    if api_key:
        querystring['api_key'] = api_key
    if group_name:
        querystring['group_name'] = group_name
    if timestamp:
        querystring['timestamp'] = timestamp
    if api:
        querystring['api'] = api
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_for_modifying_group_information(timestamp: str, api_key: str, group_name: str, api: str, cert_key: str, group_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access to modifying location object's group information
		
		required parameter : api (api.common.group.set.modify)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {'timestamp': timestamp, 'api_key': api_key, 'group_name': group_name, 'api': api, 'cert_key': cert_key, 'group_key': group_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_for_removing_group_information(api: str, api_key: str, cert_key: str, group_key: str, timestamp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access to remove location object's group information.
		
		required parameter : api (api.common.group.set.delete)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {'api': api, 'api_key': api_key, 'cert_key': cert_key, 'group_key': group_key, 'timestamp': timestamp, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_for_adding_group_memeber(cert_key: str, api_key: str, timestamp: str, group_key: str, api: str, member_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access to add location object's group member.
		
		required parameter : api (api.common.group.set.object.add)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {'cert_key': cert_key, 'api_key': api_key, 'timestamp': timestamp, 'group_key': group_key, 'api': api, 'member_key': member_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_management_api_access_to_remove_group_member(group_key: str, cert_key: str, timestamp: str, api_key: str, api: str, member_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API access to remove location object's group member.
		
		required parameter : api (api.common.group.set.object.delete)"
    
    """
    url = f"https://catchloc.p.rapidapi.com/api.partner.common.php"
    querystring = {'group_key': group_key, 'cert_key': cert_key, 'timestamp': timestamp, 'api_key': api_key, 'api': api, 'member_key': member_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "catchloc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

