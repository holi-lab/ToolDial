import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def changehistory_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A ChangeHistory item representing changes to an object in the database."
    
    """
    url = f"https://grcschema.p.rapidapi.com/ChangeHistory/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def organization_provider_data_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides clearbit data for an organization.  This can be used client side after validation and sent back in through a PATCH."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Organization/{is_id}/provider"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_provider_data_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets clearbit provider data for a Person."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Person/{is_id}/provider"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def changehistory_list(endpoint: str=None, before: str=None, team_id: str=None, object_id: str=None, type: str=None, limit: str=None, search: str=None, organization_id: str=None, schema: str=None, after: str=None, offset: str=None, person_id: str=None, sort_dir: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of ChangeHistory items representing changes to objects in the database."
    
    """
    url = f"https://grcschema.p.rapidapi.com/ChangeHistory"
    querystring = {}
    if endpoint:
        querystring['endpoint'] = endpoint
    if before:
        querystring['before'] = before
    if team_id:
        querystring['team_id'] = team_id
    if object_id:
        querystring['object_id'] = object_id
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if organization_id:
        querystring['organization_id'] = organization_id
    if schema:
        querystring['schema'] = schema
    if after:
        querystring['after'] = after
    if offset:
        querystring['offset'] = offset
    if person_id:
        querystring['person_id'] = person_id
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_list(search: str=None, country_id: int=None, offset: str=None, sort_dir: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list State objects - filterable."
    
    """
    url = f"https://grcschema.p.rapidapi.com/State"
    querystring = {}
    if search:
        querystring['search'] = search
    if country_id:
        querystring['country_id'] = country_id
    if offset:
        querystring['offset'] = offset
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_list(limit: int=None, country_id: int=None, offset: str=None, sort_dir: str=None, state_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list State objects - filterable."
    
    """
    url = f"https://grcschema.p.rapidapi.com/City"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if country_id:
        querystring['country_id'] = country_id
    if offset:
        querystring['offset'] = offset
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if state_id:
        querystring['state_id'] = state_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def role_list(search: str=None, sort_dir: str=None, name: str=None, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of all Role objects."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Role"
    querystring = {}
    if search:
        querystring['search'] = search
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_list(limit: int=None, sort_dir: str=None, name: str=None, search: str=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of all Currency objects."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Currency"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language_list(sort_dir: str=None, limit: int=None, search: str=None, offset: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of Language objects ordered by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Language"
    querystring = {}
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Language object based on the id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Language/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_list(sort_dir: str='0,1', offset: str=None, search: str=None, name: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of all Country objects."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Country"
    querystring = {}
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Country object based on the id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Country/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def organization_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an Organization object based on the id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Organization/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_list(sort_dir: str=None, offset: int=None, limit: int=None, search: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of Group objects."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Group"
    querystring = {}
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def initiative_list(name: str=None, limit: int=None, offset: int=None, search: str=None, sort_dir: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of Initiative objects."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Initiative"
    querystring = {}
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def initiative_get(is_id: int, sort_dir: str=None, name: str=None, limit: int=None, offset: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an Initiative object."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Initiative/{is_id}"
    querystring = {}
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Group object based on the id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Group/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Person object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Person/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_list(sort_dir: str=None, first_name: str=None, email: str=None, search: str=None, limit: int=None, offset: int=None, last_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of Person objects. Parameters are optional to provide filtering and pagination capability."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Person"
    querystring = {}
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if first_name:
        querystring['first_name'] = first_name
    if email:
        querystring['email'] = email
    if search:
        querystring['search'] = search
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if last_name:
        querystring['last_name'] = last_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nameprefix_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a NamePrefix object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/NamePrefix/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nameprefix_list(abbreviation: str=None, limit: int=None, prefix: str=None, sort_dir: str=None, search: str=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of NamePrefix objects. Parameters are optional to provide filtering and pagination capability."
    
    """
    url = f"https://grcschema.p.rapidapi.com/NamePrefix"
    querystring = {}
    if abbreviation:
        querystring['abbreviation'] = abbreviation
    if limit:
        querystring['limit'] = limit
    if prefix:
        querystring['prefix'] = prefix
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def namesuffix_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a NameSuffix object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/NameSuffix/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def namesuffix_list(suffix: str=None, limit: int=None, sort_dir: str=None, offset: int=None, abbreviation: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of NameSuffix objects. Parameters are optional to provide filtering and pagination capability."
    
    """
    url = f"https://grcschema.p.rapidapi.com/NameSuffix"
    querystring = {}
    if suffix:
        querystring['suffix'] = suffix
    if limit:
        querystring['limit'] = limit
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if offset:
        querystring['offset'] = offset
    if abbreviation:
        querystring['abbreviation'] = abbreviation
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def character_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Character object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Character/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def character_list(char: str=None, sort_dir: str=None, definition: str=None, limit: int=None, offset: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Character objects. Parameters are optional to provide filtering and pagination capability."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Character"
    querystring = {}
    if char:
        querystring['char'] = char
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if definition:
        querystring['definition'] = definition
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def organization_list(search: str=None, limit: str=None, offset: str=None, name: str=None, sort_dir: str=None, domain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of Organization objects ordered by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Organization"
    querystring = {}
    if search:
        querystring['search'] = search
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adsubjectmatters_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Authority Document Hierarchy"
    
    """
    url = f"https://grcschema.p.rapidapi.com/ADSubjectMatters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adrequest_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Authority Document Request List"
    
    """
    url = f"https://grcschema.p.rapidapi.com/ADRequest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authoritydocument_list(sm_id: int=None, search: str=None, parent_id: int=None, name: str=None, offset: str=None, sort_dir: str=None, limit: str=None, category_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Stub list of Authority Documents objects."
    
    """
    url = f"https://grcschema.p.rapidapi.com/AuthorityDocument"
    querystring = {}
    if sm_id:
        querystring['sm_id'] = sm_id
    if search:
        querystring['search'] = search
    if parent_id:
        querystring['parent_id'] = parent_id
    if name:
        querystring['name'] = name
    if offset:
        querystring['offset'] = offset
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if limit:
        querystring['limit'] = limit
    if category_id:
        querystring['category_id'] = category_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authoritydocument_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an Authority Document object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/AuthorityDocument/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adhierarchy_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Authority Document Hierarchy List"
    
    """
    url = f"https://grcschema.p.rapidapi.com/ADHierarchy"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subjectmatters_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of only the SubjectMatter Stub Objects"
    
    """
    url = f"https://grcschema.p.rapidapi.com/SubjectMatters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def worldcatidentities_get_list(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a grcschema object transformation of the WorldCat Identities response."
    name: WorldCat Identities fullName object.
        
    """
    url = f"https://grcschema.p.rapidapi.com/WorldCatIdentity"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a City object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/City/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a State object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/State/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viaf_autosuggest_get_list(object: str, name: str, object_id: str, limit: int=None, offset: int=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The WorlCat VIAF endpoint provides: "Access to linked names for the same entity across the world's major name authority files, including national and regional variations in language, character set, and spelling." This endpoint is a grcschema formatted example of the AutoSuggest response."
    
    """
    url = f"https://grcschema.p.rapidapi.com/WorldCatViaf"
    querystring = {'object': object, 'name': name, 'object_id': object_id, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def organization_advanced_search(search: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides an Organization stub list after applying advanced search rules."
    
    """
    url = f"https://grcschema.p.rapidapi.com/search/Organization"
    querystring = {'search': search, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_advanced_search(search: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a Group stub list after applying advanced search rules."
    
    """
    url = f"https://grcschema.p.rapidapi.com/search/Group"
    querystring = {'search': search, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def initiative_advanced_search(search: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a Initiative stub list after applying advanced search rules."
    
    """
    url = f"https://grcschema.p.rapidapi.com/search/Initiative"
    querystring = {'search': search, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_advanced_search(search: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a Person stub list after applying advanced search rules."
    
    """
    url = f"https://grcschema.p.rapidapi.com/search/Person"
    querystring = {'search': search, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authoritydocument_advanced_search(search: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides an AuthorityDocument stub list after applying advanced search rules."
    
    """
    url = f"https://grcschema.p.rapidapi.com/search/AuthorityDocument"
    querystring = {'search': search, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def knowledge_list(sort_dir: str=None, limit: int=None, offset: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Ability objects. Parameters are optional to provide filtering and pagination capability."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Knowledge"
    querystring = {}
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def knowledge_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Knowledge object."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Knowledge/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def skill_list(search: str=None, offset: int=None, sort_dir: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Skill objects. Parameters are optional to provide filtering and pagination capability."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Skill"
    querystring = {}
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def skill_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Skill object."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Skill/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ability_list(offset: int=None, search: str=None, sort_dir: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Ability objects. Parameters are optional to provide filtering and pagination capability."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Ability"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    if sort_dir:
        querystring['sort_dir'] = sort_dir
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ability_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an Ability object."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Ability/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Currency object based on the id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Currency/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def role(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Role object by id."
    
    """
    url = f"https://grcschema.p.rapidapi.com/Role/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grcschema.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

