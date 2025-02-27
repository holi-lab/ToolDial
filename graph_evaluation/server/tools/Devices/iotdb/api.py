import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def readsingleitemshardwarehardware(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_hardware item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_hardware/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfile(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single file by unique identifier."
    id: Unique identifier for the object.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/files/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsfunctionsfunctions(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single functions_functions item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/functions_functions/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwarevideos(fields: str='[]', search: str=None, sort: str='[]', offset: int=None, meta: str=None, filter: str='[]', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_videos items."
    fields: Control what fields are being returned in the object.
        search: Filter by items that contain the given search query in one of their fields.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        filter: Select items in collection by given conditions.
        limit: A limit on the number of objects that are returned.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_videos"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if search:
        querystring['search'] = search
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsconfigurations(fields: str='[]', offset: int=None, search: str=None, filter: str='[]', limit: int=None, meta: str=None, sort: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the configurations items."
    fields: Control what fields are being returned in the object.
        offset: How many items to skip when fetching data.
        search: Filter by items that contain the given search query in one of their fields.
        filter: Select items in collection by given conditions.
        limit: A limit on the number of objects that are returned.
        meta: What metadata to return in the response.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        
    """
    url = f"https://iotdb.p.rapidapi.com/items/configurations"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    if meta:
        querystring['meta'] = meta
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwarepowertypes2(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_power_types_2 item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_power_types_2/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsbatterytypes(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single battery_types item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/battery_types/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsexternalsources(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single external_sources item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/external_sources/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsbatterytypes(fields: str='[]', offset: int=None, limit: int=None, search: str=None, filter: str='[]', sort: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the battery_types items."
    fields: Control what fields are being returned in the object.
        offset: How many items to skip when fetching data.
        limit: A limit on the number of objects that are returned.
        search: Filter by items that contain the given search query in one of their fields.
        filter: Select items in collection by given conditions.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/battery_types"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if filter:
        querystring['filter'] = filter
    if sort:
        querystring['sort'] = sort
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwareconnectivity(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_connectivity item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_connectivity/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemscompany(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single company item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/company/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwarevoiceassistants(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_voice_assistants item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_voice_assistants/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwareapps(offset: int=None, limit: int=None, sort: str='[]', search: str=None, fields: str='[]', meta: str=None, filter: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_apps items."
    offset: How many items to skip when fetching data.
        limit: A limit on the number of objects that are returned.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        search: Filter by items that contain the given search query in one of their fields.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        filter: Select items in collection by given conditions.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_apps"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if search:
        querystring['search'] = search
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsfunctions(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single functions item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/functions/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsconfigurations(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single configurations item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/configurations/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwarefiles(limit: int=None, meta: str=None, fields: str='[]', filter: str='[]', sort: str='[]', offset: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_files items."
    limit: A limit on the number of objects that are returned.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        filter: Select items in collection by given conditions.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        offset: How many items to skip when fetching data.
        search: Filter by items that contain the given search query in one of their fields.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_files"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    if filter:
        querystring['filter'] = filter
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsedits(offset: int=None, meta: str=None, fields: str='[]', limit: int=None, filter: str='[]', search: str=None, sort: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the edits items."
    offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        limit: A limit on the number of objects that are returned.
        filter: Select items in collection by given conditions.
        search: Filter by items that contain the given search query in one of their fields.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        
    """
    url = f"https://iotdb.p.rapidapi.com/items/edits"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if filter:
        querystring['filter'] = filter
    if search:
        querystring['search'] = search
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwarefiles(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_files item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_files/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemscompany(search: str=None, offset: int=None, filter: str='[]', fields: str='[]', sort: str='[]', limit: int=None, meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the company items."
    search: Filter by items that contain the given search query in one of their fields.
        offset: How many items to skip when fetching data.
        filter: Select items in collection by given conditions.
        fields: Control what fields are being returned in the object.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        limit: A limit on the number of objects that are returned.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/company"
    querystring = {}
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    if filter:
        querystring['filter'] = filter
    if fields:
        querystring['fields'] = fields
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwaresensors(limit: int=None, sort: str='[]', search: str=None, filter: str='[]', offset: int=None, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_sensors items."
    limit: A limit on the number of objects that are returned.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        search: Filter by items that contain the given search query in one of their fields.
        filter: Select items in collection by given conditions.
        offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_sensors"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if search:
        querystring['search'] = search
    if filter:
        querystring['filter'] = filter
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsedits(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single edits item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/edits/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsconnectivity(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single connectivity item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/connectivity/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsvoiceassistants(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single voice_assistants item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/voice_assistants/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwareservices(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_services item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_services/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwarepowertypes2(fields: str='[]', limit: int=None, offset: int=None, search: str=None, filter: str='[]', sort: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_power_types_2 items."
    fields: Control what fields are being returned in the object.
        limit: A limit on the number of objects that are returned.
        offset: How many items to skip when fetching data.
        search: Filter by items that contain the given search query in one of their fields.
        filter: Select items in collection by given conditions.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_power_types_2"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    if filter:
        querystring['filter'] = filter
    if sort:
        querystring['sort'] = sort
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsexternalitems(sort: str='[]', offset: int=None, fields: str='[]', filter: str='[]', limit: int=None, meta: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the external_items items."
    sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        offset: How many items to skip when fetching data.
        fields: Control what fields are being returned in the object.
        filter: Select items in collection by given conditions.
        limit: A limit on the number of objects that are returned.
        meta: What metadata to return in the response.
        search: Filter by items that contain the given search query in one of their fields.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/external_items"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if fields:
        querystring['fields'] = fields
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    if meta:
        querystring['meta'] = meta
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwarehardware(limit: int=None, fields: str='[]', filter: str='[]', sort: str='[]', meta: str=None, offset: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_hardware items."
    limit: A limit on the number of objects that are returned.
        fields: Control what fields are being returned in the object.
        filter: Select items in collection by given conditions.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        meta: What metadata to return in the response.
        offset: How many items to skip when fetching data.
        search: Filter by items that contain the given search query in one of their fields.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_hardware"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    if filter:
        querystring['filter'] = filter
    if sort:
        querystring['sort'] = sort
    if meta:
        querystring['meta'] = meta
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsexternalitems(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single external_items item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/external_items/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardware(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsvideos(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single videos item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/videos/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwarefunctions(filter: str='[]', offset: int=None, meta: str=None, fields: str='[]', search: str=None, sort: str='[]', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_functions items."
    filter: Select items in collection by given conditions.
        offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        search: Filter by items that contain the given search query in one of their fields.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        limit: A limit on the number of objects that are returned.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_functions"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    if search:
        querystring['search'] = search
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwareconnectivity(offset: int=None, meta: str=None, fields: str='[]', sort: str='[]', filter: str='[]', limit: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_connectivity items."
    offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        filter: Select items in collection by given conditions.
        limit: A limit on the number of objects that are returned.
        search: Filter by items that contain the given search query in one of their fields.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_connectivity"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    if sort:
        querystring['sort'] = sort
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsfunctions(search: str=None, limit: int=None, filter: str='[]', fields: str='[]', meta: str=None, sort: str='[]', offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the functions items."
    search: Filter by items that contain the given search query in one of their fields.
        limit: A limit on the number of objects that are returned.
        filter: Select items in collection by given conditions.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        offset: How many items to skip when fetching data.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/functions"
    querystring = {}
    if search:
        querystring['search'] = search
    if limit:
        querystring['limit'] = limit
    if filter:
        querystring['filter'] = filter
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardware(offset: int=None, search: str=None, sort: str='[]', filter: str='[]', limit: int=None, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware items."
    offset: How many items to skip when fetching data.
        search: Filter by items that contain the given search query in one of their fields.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        filter: Select items in collection by given conditions.
        limit: A limit on the number of objects that are returned.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    if sort:
        querystring['sort'] = sort
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwareservices(filter: str='[]', offset: int=None, limit: int=None, search: str=None, fields: str='[]', sort: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_services items."
    filter: Select items in collection by given conditions.
        offset: How many items to skip when fetching data.
        limit: A limit on the number of objects that are returned.
        search: Filter by items that contain the given search query in one of their fields.
        fields: Control what fields are being returned in the object.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_services"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if fields:
        querystring['fields'] = fields
    if sort:
        querystring['sort'] = sort
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsconnectivity(meta: str=None, filter: str='[]', fields: str='[]', search: str=None, offset: int=None, sort: str='[]', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the connectivity items."
    meta: What metadata to return in the response.
        filter: Select items in collection by given conditions.
        fields: Control what fields are being returned in the object.
        search: Filter by items that contain the given search query in one of their fields.
        offset: How many items to skip when fetching data.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        limit: A limit on the number of objects that are returned.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/connectivity"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if filter:
        querystring['filter'] = filter
    if fields:
        querystring['fields'] = fields
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwaresensors(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_sensors item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_sensors/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemspowertypes(limit: int=None, meta: str=None, offset: int=None, sort: str='[]', filter: str='[]', fields: str='[]', search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the power_types items."
    limit: A limit on the number of objects that are returned.
        meta: What metadata to return in the response.
        offset: How many items to skip when fetching data.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        filter: Select items in collection by given conditions.
        fields: Control what fields are being returned in the object.
        search: Filter by items that contain the given search query in one of their fields.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/power_types"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if meta:
        querystring['meta'] = meta
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if filter:
        querystring['filter'] = filter
    if fields:
        querystring['fields'] = fields
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwareapps(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_apps item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_apps/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsapps(limit: int=None, search: str=None, offset: int=None, filter: str='[]', fields: str='[]', meta: str=None, sort: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the apps items."
    limit: A limit on the number of objects that are returned.
        search: Filter by items that contain the given search query in one of their fields.
        offset: How many items to skip when fetching data.
        filter: Select items in collection by given conditions.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        
    """
    url = f"https://iotdb.p.rapidapi.com/items/apps"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    if filter:
        querystring['filter'] = filter
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemshardwarevoiceassistants(search: str=None, limit: int=None, filter: str='[]', offset: int=None, meta: str=None, fields: str='[]', sort: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the hardware_voice_assistants items."
    search: Filter by items that contain the given search query in one of their fields.
        limit: A limit on the number of objects that are returned.
        filter: Select items in collection by given conditions.
        offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_voice_assistants"
    querystring = {}
    if search:
        querystring['search'] = search
    if limit:
        querystring['limit'] = limit
    if filter:
        querystring['filter'] = filter
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwarefunctions(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_functions item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_functions/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsbrand(meta: str=None, fields: str='[]', search: str=None, limit: int=None, sort: str='[]', filter: str='[]', offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the brand items."
    meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        search: Filter by items that contain the given search query in one of their fields.
        limit: A limit on the number of objects that are returned.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        filter: Select items in collection by given conditions.
        offset: How many items to skip when fetching data.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/brand"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    if search:
        querystring['search'] = search
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if filter:
        querystring['filter'] = filter
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsservices(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single services item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/services/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsvoiceassistants(fields: str='[]', filter: str='[]', search: str=None, meta: str=None, offset: int=None, sort: str='[]', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the voice_assistants items."
    fields: Control what fields are being returned in the object.
        filter: Select items in collection by given conditions.
        search: Filter by items that contain the given search query in one of their fields.
        meta: What metadata to return in the response.
        offset: How many items to skip when fetching data.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        limit: A limit on the number of objects that are returned.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/voice_assistants"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if filter:
        querystring['filter'] = filter
    if search:
        querystring['search'] = search
    if meta:
        querystring['meta'] = meta
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsapps(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single apps item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/apps/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsbrand(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single brand item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/brand/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsexternalsources(filter: str='[]', search: str=None, sort: str='[]', fields: str='[]', limit: int=None, offset: int=None, meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the external_sources items."
    filter: Select items in collection by given conditions.
        search: Filter by items that contain the given search query in one of their fields.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        fields: Control what fields are being returned in the object.
        limit: A limit on the number of objects that are returned.
        offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/external_sources"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if search:
        querystring['search'] = search
    if sort:
        querystring['sort'] = sort
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsvideos(filter: str='[]', search: str=None, offset: int=None, meta: str=None, limit: int=None, fields: str='[]', sort: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the videos items."
    filter: Select items in collection by given conditions.
        search: Filter by items that contain the given search query in one of their fields.
        offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        limit: A limit on the number of objects that are returned.
        fields: Control what fields are being returned in the object.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        
    """
    url = f"https://iotdb.p.rapidapi.com/items/videos"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsservices(offset: int=None, limit: int=None, filter: str='[]', meta: str=None, sort: str='[]', fields: str='[]', search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the services items."
    offset: How many items to skip when fetching data.
        limit: A limit on the number of objects that are returned.
        filter: Select items in collection by given conditions.
        meta: What metadata to return in the response.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        fields: Control what fields are being returned in the object.
        search: Filter by items that contain the given search query in one of their fields.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/services"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if filter:
        querystring['filter'] = filter
    if meta:
        querystring['meta'] = meta
    if sort:
        querystring['sort'] = sort
    if fields:
        querystring['fields'] = fields
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemssensors(offset: int=None, meta: str=None, fields: str='[]', sort: str='[]', search: str=None, filter: str='[]', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the sensors items."
    offset: How many items to skip when fetching data.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        search: Filter by items that contain the given search query in one of their fields.
        filter: Select items in collection by given conditions.
        limit: A limit on the number of objects that are returned.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/sensors"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    if sort:
        querystring['sort'] = sort
    if search:
        querystring['search'] = search
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemshardwarevideos(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single hardware_videos item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/hardware_videos/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsfunctionsfunctions(fields: str='[]', limit: int=None, search: str=None, offset: int=None, sort: str='[]', meta: str=None, filter: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the functions_functions items."
    fields: Control what fields are being returned in the object.
        limit: A limit on the number of objects that are returned.
        search: Filter by items that contain the given search query in one of their fields.
        offset: How many items to skip when fetching data.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        meta: What metadata to return in the response.
        filter: Select items in collection by given conditions.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/functions_functions"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if meta:
        querystring['meta'] = meta
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemsusbtypes(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single usb_types item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/usb_types/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemspowertypes(is_id: str, fields: str='[]', meta: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single power_types item by unique identifier."
    id: Index of the item.
        fields: Control what fields are being returned in the object.
        meta: What metadata to return in the response.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/power_types/{is_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if meta:
        querystring['meta'] = meta
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readitemsusbtypes(meta: str=None, search: str=None, filter: str='[]', sort: str='[]', fields: str='[]', limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the usb_types items."
    meta: What metadata to return in the response.
        search: Filter by items that contain the given search query in one of their fields.
        filter: Select items in collection by given conditions.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        fields: Control what fields are being returned in the object.
        limit: A limit on the number of objects that are returned.
        offset: How many items to skip when fetching data.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/usb_types"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if search:
        querystring['search'] = search
    if filter:
        querystring['filter'] = filter
    if sort:
        querystring['sort'] = sort
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def readsingleitemssensors(is_id: str, meta: str=None, fields: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single sensors item by unique identifier."
    id: Index of the item.
        meta: What metadata to return in the response.
        fields: Control what fields are being returned in the object.
        
    """
    url = f"https://iotdb.p.rapidapi.com/items/sensors/{is_id}"
    querystring = {}
    if meta:
        querystring['meta'] = meta
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfiles(filter: str='[]', meta: str=None, offset: int=None, fields: str='[]', search: str=None, sort: str='[]', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the files."
    filter: Select items in collection by given conditions.
        meta: What metadata to return in the response.
        offset: How many items to skip when fetching data.
        fields: Control what fields are being returned in the object.
        search: Filter by items that contain the given search query in one of their fields.
        sort: How to sort the returned items. `sort` is a CSV of fields used to sort the fetched items. Sorting defaults to ascending (ASC) order but a minus sign (` - `) can be used to reverse this to descending (DESC) order. Fields are prioritized by their order in the CSV. You can also use a ` ? ` to sort randomly.

        limit: A limit on the number of objects that are returned.
        
    """
    url = f"https://iotdb.p.rapidapi.com/files"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if meta:
        querystring['meta'] = meta
    if offset:
        querystring['offset'] = offset
    if fields:
        querystring['fields'] = fields
    if search:
        querystring['search'] = search
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iotdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

