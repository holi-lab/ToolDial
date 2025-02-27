import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_datapoint_by_id(datapoint_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datapoint details"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/datapoints/{datapoint_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def click_stream(group: str='147609', datapoint: str='1730798', withconversion: str=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the clickstream for the desired object."
    group: Id of the group. Mutually exclusive with datapoint.
        datapoint: Id of the datapoint. Mutually exclusive with group.
        withconversion: Add conversion data to clickstream.
        fields: Fields to retrieve for the clickstream. Comma separated values.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/clickstream"
    querystring = {}
    if group:
        querystring['group'] = group
    if datapoint:
        querystring['datapoint'] = datapoint
    if withconversion:
        querystring['withConversion'] = withconversion
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hit_list(fromday: str=None, today: str=None, offset: str=None, limit: str=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A detailed hits list for the last 90 days maximum"
    fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        offset: Row key from which to start retrieve objects. Use the lastKey in object response.
        limit: Maximum elements to retrieve. Defaults to 20 if not specified.
        fields: Fields to retrieve for the hit list. Comma separated values.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/hits"
    querystring = {}
    if fromday:
        querystring['fromDay'] = fromday
    if today:
        querystring['toDay'] = today
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_count(name: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a count of domains available"
    name: A pattern name expression. Example: twit*
        type: A pattern expression for type. It can be "system", "dedicated", "go", "personal". Default: system. Example: system
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/domains/count"
    querystring = {}
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tag(tag_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the specified tag."
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/tags/{tag_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groups_count(status: str=None, tags: str=None, createdafter: str=None, createdbefore: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Groups count"
    status: Filter by group status. It can be "active", "deleted", "all". Default is "all"
        tags: A comma separated list oftags you want to filter with.
        createdafter: A date in the format YYYYMMDDHHmm.
        createdbefore: A date in the format YYYYMMDDHHmm.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/groups/count"
    querystring = {}
    if status:
        querystring['status'] = status
    if tags:
        querystring['tags'] = tags
    if createdafter:
        querystring['createdafter'] = createdafter
    if createdbefore:
        querystring['createdbefore'] = createdbefore
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_deail(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of the current logged account."
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def guest_detail(guestid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Guest detail. Access only for master key."
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/account/guests/{guestid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_domain_by_id(domain_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the detail about a domain"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/domains/{domain_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groups_hits(group_id: str, fromday: str=None, today: str=None, offset: str=None, limit: str=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Groups Hits"
    fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        offset: Row key from which to start retrieve objects. Use the lastKey in object response.
        limit: Maximum elements to retrieve. Defaults to 20 if not specified.
        fields: Fields to retrieve for the hit list. Comma separated values.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/groups/{group_id}/hits"
    querystring = {}
    if fromday:
        querystring['fromDay'] = fromday
    if today:
        querystring['toDay'] = today
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_conversion_by_id(conversion_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get conversion by id"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/conversions/{conversion_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def guest_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Guest list account. Access only for master key."
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/account/guests"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_group_by_id(group_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get group by id"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/groups/{group_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datapoint_hit_list(datapoint_id: str, fromday: str=None, today: str=None, offset: str=None, limit: str=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A detailed hits list for the last 90 days maximum by DataPoint"
    fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        offset: Row key from which to start retrieve objects. Use the last Key in object response.
        limit: Maximum elements to retrieve. Defaults to 20 if not specified.
        fields: Fields to retrieve for the hit list. Comma separated values.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/datapoints/{datapoint_id}/hits"
    querystring = {}
    if fromday:
        querystring['fromDay'] = fromday
    if today:
        querystring['toDay'] = today
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_full_list_of_datapoints_with_statistics(timeframe: str, type: str, today: str=None, status: str=None, tag: str=None, favourite: bool=None, sortby: str=None, sortdirection: str=None, limit: str=None, fromday: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a full list of datapoints with statistics, based on filters"
    timeframe: Timeframe can be "today", "yesterday", "last7", "last30", "last90", "beginning", "custom". If "custom" use also fromDay-toDay parameters.
        type: Type of datapoint, "tl" or "tp".
        today: A date in the format YYYYMMDDHHmm.
        status: By default "all": a filter by entity status (deleted, paused, spam, active).
        tag: Name of tag to filter with.
        favourite: Filter by favourites only.
        sortby: Sort list by specified field (totalclicks, totalviews, uniqueclicks, uniqueviews, conversionsvalue, convertedclicks, spiderhitscount, entityData.datapointname, entityData.creationdate).
        sortdirection: Direction to sort with: "asc" or "desc".
        limit: Maximum elements to retrieve. Default is 20.
        fromday: A date in the format YYYYMMDDHHmm.
        offset: Where to start when retrieving data. 0 if it's not specified.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/aggregated/summary/datapoints"
    querystring = {'timeframe': timeframe, 'type': type, }
    if today:
        querystring['toDay'] = today
    if status:
        querystring['status'] = status
    if tag:
        querystring['tag'] = tag
    if favourite:
        querystring['favourite'] = favourite
    if sortby:
        querystring['sortBy'] = sortby
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    if limit:
        querystring['limit'] = limit
    if fromday:
        querystring['fromday'] = fromday
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datapoints_count(type: str=None, tags: str=None, createdafter: str=None, createdbefore: str=None, status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datapoints count"
    type: The type of datapoint to retrieve. Possible types: TL, TP, TD.
        tags: A comma separated list of tags you want to filter with.
        createdafter: A date in the format YYYYMMDDHHmm.
        createdbefore: A date in the format YYYYMMDDHHmm.
        status: Filter by datapoint status. It can be "active", "deleted", "spam", "paused", "all". Default is "all".
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/datapoints/count"
    querystring = {}
    if type:
        querystring['type'] = type
    if tags:
        querystring['tags'] = tags
    if createdafter:
        querystring['createdafter'] = createdafter
    if createdbefore:
        querystring['createdbefore'] = createdbefore
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversions_count(status: str=None, createdafter: str=None, createdbefore: str=None, textsearch: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Conversions count"
    status: Filter by group status. It can be "active", "deleted", "all". Default is "all"
        createdafter: A date in the format YYYYMMDDHHmm.
        createdbefore: A date in the format YYYYMMDDHHmm.
        textsearch: A pattern name expression. Example: "twit" will match with all the conversion names starting with "twit"
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/conversions/count"
    querystring = {}
    if status:
        querystring['status'] = status
    if createdafter:
        querystring['createdafter'] = createdafter
    if createdbefore:
        querystring['createdbefore'] = createdbefore
    if textsearch:
        querystring['textsearch'] = textsearch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def plan_detail(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail of the plan used by this account. Valid only for MASTER key."
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/account/plan"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def guest_permissions(guestid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of permission that guest have"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/account/guests/{guestid}/permissions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datapoint_health_statuses_count(httpcode: str=None, isspam: str=None, latency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datapoint health statuses Count"
    httpcode: Count only statuses with the given HTTP response code. It can be 2XX, 4XX, 5XX.
        isspam: Count only statuses with the given isSpam value.
        latency: Count only statuses with latency >= the given value.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/datapoints/health/count"
    querystring = {}
    if httpcode:
        querystring['httpcode'] = httpcode
    if isspam:
        querystring['isspam'] = isspam
    if latency:
        querystring['latency'] = latency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_full_list_of_groups_with_statistics(timeframe: str, fromday: str=None, today: str=None, status: str=None, favourite: bool=None, sortdirection: str=None, offset: str=None, tag: str=None, sortby: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a full list of groups with statistics"
    timeframe: Timeframe can be "today", "yesterday", "last7", "last30", "last90", "beginning", "custom". If "custom" use also fromDay-toDay parameters.
        fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        status: By default "all": a filter by group status (deleted, active).
        favourite: Filter by favourites only.
        sortdirection: Direction to sort with, "asc" or "desc".
        offset: Where to start when retrieving data. 0 if it's not specified.
        tag: Name of tag to filter with.
        sortby: Sort list by specified field (totalclicks, totalviews, uniqueclicks, uniqueviews, conversionsvalue, convertedclicks, spiderhitscount, entityData.groupname, entityData.creationdate).
        limit: Maximum elements to retrieve. Defaults is 20.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/aggregated/summary/groups"
    querystring = {'timeframe': timeframe, }
    if fromday:
        querystring['fromDay'] = fromday
    if today:
        querystring['toDay'] = today
    if status:
        querystring['status'] = status
    if favourite:
        querystring['favourite'] = favourite
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    if offset:
        querystring['offset'] = offset
    if tag:
        querystring['tag'] = tag
    if sortby:
        querystring['sortby'] = sortby
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all the tags associated to the user."
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_groups(status: str=None, tags: str=None, createdafter: str=None, createdbefore: str=None, limit: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List groups"
    status: It can be "deleted", "active" or "all". Default is "all".
        tags: A comma separated list oftags you want to filter with.
        createdafter: A date in the format YYYYMMDDHHmm.
        createdbefore: A date in the format YYYYMMDDHHmm.
        limit: Maximum elements to retrieve. 20 if not specified.
        offset: Where to start when retrieving domains. Zero if not specified.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/groups"
    querystring = {}
    if status:
        querystring['status'] = status
    if tags:
        querystring['tags'] = tags
    if createdafter:
        querystring['createdAfter'] = createdafter
    if createdbefore:
        querystring['createdBefore'] = createdbefore
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datapoint_health_status_by_id(datapoint_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve some informations about the status of the destination URL associated to a datapoint. Available info depends on the account plan"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/datapoints/{datapoint_id}/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_aggregated_counts_with_daily_breakdown(timeframe: str, fromday: str=None, today: str=None, groupby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get aggregated counts with daily breakdown"
    timeframe: Timeframe can be "today", "yesterday", "last7", "last30", "last90", "beginning", "custom". If "custom" use also fromDay-toDay parameters.
        fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        groupby: Can select the granularity of the list “day, week, month”. BEWARE in “beginning” timeframe you must set this to “month”.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/aggregated/list"
    querystring = {'timeframe': timeframe, }
    if fromday:
        querystring['fromDay'] = fromday
    if today:
        querystring['toDay'] = today
    if groupby:
        querystring['groupBy'] = groupby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_domains(name: str=None, type: str=None, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List domains to which is possible to register a datapoint"
    name: A pattern name expression. Example: twit*
        type: A pattern expression for type. It can be "system", "dedicated", "go", "personal". Default: system. Example: system
        offset: Where to start when retrieving domains. Zero if not specified.
        limit: Maximum elements to retrieve. 20 if not specified.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/domains"
    querystring = {}
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_datapoint_health_statuses(httpcode: str=None, isspam: bool=None, latency: str=None, limit: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the list of the links which have some status information attached."
    httpcode: Filter statuses by HTTP response code to a simple GET request. It can be 2XX, 4XX, 5XX.
        isspam: If true, it returns only the datapoints with a blacklisted destination URL.
        latency: Retrieve all the datapoint with latency (milliseconds) >= the specified value. Example: latency = 2500 , which means latency >= 2,5 seconds.
        limit: Maximum elements to retrieve. 20 if not specified.
        offset: Where to start when retrieving domains. Zero if not specified.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/datapoints/health"
    querystring = {}
    if httpcode:
        querystring['httpcode'] = httpcode
    if isspam:
        querystring['isspam'] = isspam
    if latency:
        querystring['latency'] = latency
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_conversions(status: str=None, createdafter: str=None, createdbefore: str=None, name: str=None, limit: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List conversions"
    status: It can be "deleted", "active" or "all". Default is "all".
        createdafter: A date in the format YYYYMMDDHHmm.
        createdbefore: A date in the format YYYYMMDDHHmm.
        name: A pattern name expression. Example: twit*
        limit: Maximum elements to retrieve. 20 if not specified.
        offset: Where to start when retrieving domains. Zero if not specified.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/conversions"
    querystring = {}
    if status:
        querystring['status'] = status
    if createdafter:
        querystring['createdafter'] = createdafter
    if createdbefore:
        querystring['createdbefore'] = createdbefore
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversions_hits(conversion_id: str, fields: str=None, fromday: str=None, today: str=None, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A detailed hits list for the last 90 days maximum by conversion"
    fields: Fields to retrieve for the hit list. Comma separated values.
        fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        limit: Maximum elements to retrieve. Defaults to 20 if not specified.
        offset: Row key from which to start retrieve objects. Use the lastKey in object response.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/conversions/{conversion_id}/hits"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if fromday:
        querystring['fromday'] = fromday
    if today:
        querystring['today'] = today
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datapoints_by_group(group_id: str, offset: str=None, limit: str=None, type: str=None, status: str=None, tags: str=None, createdafter: str=None, createdbefore: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datapoints by group"
    offset: Where to start when retrieving domains. Zero if not specified.
        limit: Maximum elements to retrieve. 20 if not specified.
        type: The type of datapoint to retrieve. Possible types: TL, TP, TD.
        status: Filter by datapoint status. It can be "active", "deleted", "spam", "paused", "all". Default is "all".
        tags: A comma separated list oftags you want to filter with.
        createdafter: A date in the format YYYYMMDDHHmm.
        createdbefore: A date in the format YYYYMMDDHHmm.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/groups/{group_id}/datapoints"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    if status:
        querystring['status'] = status
    if tags:
        querystring['tags'] = tags
    if createdafter:
        querystring['createdafter'] = createdafter
    if createdbefore:
        querystring['createdbefore'] = createdbefore
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_full_list_of_conversions_with_statistics(timeframe: str, fromday: str=None, today: str=None, sortdirection: str=None, status: str=None, sortby: str=None, favourite: bool=None, limit: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a full list of conversions with statistics"
    timeframe: Timeframe can be "today", "yesterday", "last7", "last30", "last90", "beginning", "custom". If "custom" use also fromDay-toDay parameters.
        fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        sortdirection: Direction to sort with, "asc" or "desc".
        status: Filter by status (all, active, deleted).
        sortby: Sort list by specified field (count, lasthitdate, entityData.name, entityData.value, entityData.creationdate).
        favourite: Filter by favourites only.
        limit: Maximum elements to retrieve. Defaults is 20.
        offset: Where to start when retrieving data. 0 if it's not specified.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/aggregated/summary/conversions"
    querystring = {'timeframe': timeframe, }
    if fromday:
        querystring['fromDay'] = fromday
    if today:
        querystring['toDay'] = today
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    if status:
        querystring['status'] = status
    if sortby:
        querystring['sortby'] = sortby
    if favourite:
        querystring['favourite'] = favourite
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datapoints_associated_to_a_tag(tag_id: str, type: str='TL', status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all the datapoints associated to the user and to the selected tag"
    type: It can be TL, TP or TD.
        status: Filter by datapoint status. It can be "active", "deleted", "spam", "paused", "all". Default is "all".
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/tags/{tag_id}/datapoints"
    querystring = {}
    if type:
        querystring['type'] = type
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datapoints_associated_to_conversion(conversion_id: str, limit: str=None, offset: str=None, status: str=None, type: str=None, tags: str=None, createdbefore: str=None, createdafter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all datapoints associated to this conversion"
    limit: Maximum elements to retrieve. 20 if not specified.
        offset: Where to start when retrieving domains. Zero if not specified.
        status: Filter by datapoint status. It can be "active", "deleted", "spam", "paused", "all". Default is "all".
        type: The type of datapoint to retrieve. Possible types: TL, TP, TD.
        tags: A comma separated list oftags you want to filter with.
        createdbefore: A date in the format YYYYMMDDHHmm.
        createdafter: A date in the format YYYYMMDDHHmm.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/conversions/{conversion_id}/datapoints"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if status:
        querystring['status'] = status
    if type:
        querystring['type'] = type
    if tags:
        querystring['tags'] = tags
    if createdbefore:
        querystring['createdBefore'] = createdbefore
    if createdafter:
        querystring['createdAfter'] = createdafter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_domains_whitelist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List domains whitelist"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/account/domainwhitelist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groups_associated_to_a_tag(tag_id: str, status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all the groups associated to the user and to the selected tag"
    status: Filter by group status. It can be "active", "deleted", "all". Default is "all"
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/tags/{tag_id}/groups"
    querystring = {}
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a count of the tags. You can use the same filters of "/tags""
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/tags/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_datapoints(limit: str=None, offset: str=None, type: str=None, status: str=None, tags: str=None, createdafter: str=None, createdbefore: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all the datapoints associated to the user"
    limit: Maximum elements to retrieve. 20 if not specified.
        offset: Where to start when retrieving datapoints. Default to 0 if not specified.
        type: The type of datapoint to retrieve. Possible types: TL, TP, TD.
        status: Filter by datapoint status. It can be "active", "deleted", "spam", "paused", "all". Default is "all".
        tags: A comma separated list of tags you want to filter with.
        createdafter: A date in the format YYYYMMDDHHmm.
        createdbefore: A date in the format YYYYMMDDHHmm.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/datapoints"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if type:
        querystring['type'] = type
    if status:
        querystring['status'] = status
    if tags:
        querystring['tags'] = tags
    if createdafter:
        querystring['createdafter'] = createdafter
    if createdbefore:
        querystring['createdbefore'] = createdbefore
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_custom_blacklisted_ips(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List custom blacklisted IPs"
    
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/account/ipblacklist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_aggregated_counts_on_customer(timeframe: str, hourly: bool=None, fromday: str=None, today: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get aggregated counts on customer"
    timeframe: Timeframe can be "today", "yesterday", "last7", "last30", "last90", "beginning", "custom". If "custom" use also fromDay-toDay parameters.
        hourly: Hourly breakdown, if selected only valid timeframe is "today".
        fromday: A date in the format YYYYMMDDHHmm.
        today: A date in the format YYYYMMDDHHmm.
        
    """
    url = f"https://ideas2it-ClickMeter-v1.p.rapidapi.com/aggregated"
    querystring = {'timeframe': timeframe, }
    if hourly:
        querystring['hourly'] = hourly
    if fromday:
        querystring['fromDay'] = fromday
    if today:
        querystring['toDay'] = today
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-ClickMeter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

