import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def events_by_multiple_filters_today(impact_levels: str, regions: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events valid to the supplied filters for today, the output is sorted by the event-time. The event-time is **utc**
		Currently you can filter by:
		
		- regions
		- impact_levels
		
		The *regions* could be a single region or multiple examples are:
		
		-  United States
		-  Euro Zone, United States
		
		*regions* is case insensitive. Valid options can be  obtained by the enpoint: `/regions`
		
		The *impact_levels* could be a single level or multiple, examples are:
		
		-  high
		-  medium, high
		- low
		
		Setting *impact_levels* and *regions* you can use **all**. If you use "all" as filter for impact level and supply regions it will result in a response with all impact levels (low, medium, high) for these regions."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/query/{impact_levels}/{regions}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_region_today(regions: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events for the requested *regions* **for today**, the output is sorted by the event-time. The event-time is **utc**. 
		The *regions* could be a single region or multiple examples are:
		-  Germany
		-  Germany, United States
		
		*regions* is case insensitive. Valid options can be  obtained by the enpoint: `/regions`"
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/region/{regions}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_impact_today(impact_levels: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events with the requested *impact_levels* **for today**, the output is sorted by the event-time. The event-time is **utc**.
		The *impact_levels* could be a single level or multiple, examples are:
		-  high
		-  medium, high
		- low"
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/impact/{impact_levels}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies_today(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available **currency options** within the selected daterange **for today**."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regions_today(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available **region options** within the selected daterange **for today**."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_currency_today(currency_abbreviation: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events for the requested *currency_abbreviation* **for today**, the output is sorted by the event-time. The event-time is **utc**. 
		*currency_abbreviation* is case insensitive. 
		
		Valid options can be  obtained by the enpoint: `/currencies`"
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/currency/{currency_abbreviation}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_today(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events of today sorted by the event-time. The event-time is **utc**."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def next_events_by_impact(impact_levels: str, num_events: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the next *num_events* events for the specified *impact_levels* sorted by event-time (in UTC).
		
		*Impact_levels* can be a single level or multiple, examples are: "high" or "medium, high"."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/impact/{impact_levels}/next/{num_events}"
    querystring = {}
    if num_events:
        querystring['num_events'] = num_events
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def next_events_by_region(regions: str, num_events: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the next *num_events* events for the specified *regions* sorted by event-time (in UTC).
		
		*Regions* can be a single region or multiple, examples are: "Germany" or "Germany, United States". The input for *regions* is case insensitive. 
		
		Valid options for *regions* can be obtained from the endpoint: `/regions`"
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/region/{regions}/next/{num_events}"
    querystring = {}
    if num_events:
        querystring['num_events'] = num_events
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(ts_end: int=1756771140000, ts_start: int=1598072400000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available **currency options** within the selected daterange.
		You can pick the daterange by supplying timestamps (in milliseconds) for *ts_start* and *ts_end*.
		If none is supplied the daterange is the current day, starting from UTC 00:00 to 23:59."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/currencies/{ts_start}/{ts_end}"
    querystring = {}
    if ts_end:
        querystring['ts_end'] = ts_end
    if ts_start:
        querystring['ts_start'] = ts_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regions(ts_start: int=1598072400000, ts_end: int=1756771140000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available **region options** within the selected daterange.
		You can pick the daterange by supplying timestamps (in milliseconds) for *ts_start* and *ts_end*.
		If none is supplied the daterange is the current day, starting from UTC 00:00 to 23:59."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/regions/{ts_start}/{ts_end}"
    querystring = {}
    if ts_start:
        querystring['ts_start'] = ts_start
    if ts_end:
        querystring['ts_end'] = ts_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_impact(impact_levels: str, ts_end: int=1756771140000, ts_start: int=1598072400000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events with the requested *impact_levels* the output is sorted by the event-time. The event-time is **utc**. You can pick a daterange by supplying timestamps ( in milliseconds) for *ts_start* and *ts_end*.
		
		If none is supplied the daterange is the current day, starting from UTC 00:00 to 23:59.
		The *impact_levels* could be a single level or multiple, examples are:
		
		-  high
		-  medium, high
		- low"
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/impact/{impact_levels}/{ts_start}/{ts_end}"
    querystring = {}
    if ts_end:
        querystring['ts_end'] = ts_end
    if ts_start:
        querystring['ts_start'] = ts_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(ts_end: int=1756771140000, ts_start: int=1598072400000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events sorted by the event-time. The event-time is **utc**. You can pick the daterange by supplying timestamps ( in milliseconds) for *ts_start*and *ts_end*.
		If none is supplied the daterange is the current day, starting from UTC 00:00 to 23:59."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/{ts_start}/{ts_end}"
    querystring = {}
    if ts_end:
        querystring['ts_end'] = ts_end
    if ts_start:
        querystring['ts_start'] = ts_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_currency(currency_abbreviation: str, ts_end: int=1756771140000, ts_start: int=1598072400000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events for the requested *currency_abbreviation* the output is sorted by the event-time. The event-time is **utc**. You can pick a daterange by supplying timestamps ( in milliseconds) for *ts_start* and *ts_end*.
		
		If none is supplied the daterange is the current day, starting from UTC 00:00 to 23:59.
		
		*currency_abbreviation* is case insensitive. Valid options can be  obtained by the enpoint: `/currencies`"
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/currency/{currency_abbreviation}/{ts_start}/{ts_end}"
    querystring = {}
    if ts_end:
        querystring['ts_end'] = ts_end
    if ts_start:
        querystring['ts_start'] = ts_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_region(regions: str, ts_end: int=1756771140000, ts_start: int=1598072400000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events for the requested *regions* the output is sorted by the event-time. The event-time is **utc**. You can pick a daterange by supplying timestamps ( in milliseconds) for *ts_start*and *ts_end*.
		
		If none is supplied the daterange is the current day, starting from UTC 00:00 to 23:59.
		The *regions* could be a single region or multiple examples are:
		
		-  Germany
		-  Germany, United States
		
		*regions* is case insensitive. Valid options can be  obtained by the enpoint: `/regions`"
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/region/{regions}/{ts_start}/{ts_end}"
    querystring = {}
    if ts_end:
        querystring['ts_end'] = ts_end
    if ts_start:
        querystring['ts_start'] = ts_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_multiple_filters(regions: str, impact_levels: str, ts_end: int=1756771140000, ts_start: int=1598072400000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events valid to the supplied filters, the output is sorted by the event-time. The event-time is **utc**. You can pick a daterange by supplying timestamps ( in milliseconds) for *ts_start* and *ts_end*.
		
		If none is supplied the daterange is the current day, starting from UTC 00:00 to 23:59. 
		Currently you can filter by:
		
		- regions
		- impact_levels
		
		The *regions* could be a single region or multiple examples are:
		
		-  United States
		-  Euro Zone, United States
		
		*regions* is case insensitive. Valid options can be  obtained by the enpoint: `/regions`
		
		The *impact_levels* could be a single level or multiple, examples are:
		
		-  high
		-  medium, high
		- low
		
		Setting *impact_levels* and *regions* you can use **all**. If you use "all" as filter for impact level and supply regions it will result in a response with all impact levels (low, medium, high) for these regions."
    
    """
    url = f"https://economiccalendar.p.rapidapi.com/events/query/{impact_levels}/{regions}/{ts_start}/{ts_end}"
    querystring = {}
    if ts_end:
        querystring['ts_end'] = ts_end
    if ts_start:
        querystring['ts_start'] = ts_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economiccalendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

