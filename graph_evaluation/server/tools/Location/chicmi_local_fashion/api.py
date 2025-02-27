import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_fashion_events_in_city(city: str, days: int=5, date_from: str=None, date_to: str=None, sectors: str=None, types: str=None, designers: str=None, stores: str=None, max_results: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fashion events in a particular city"
    city: The city to query. Values are currently "london", "glasgow", "edinburgh" or "manchester"
        days: Show events from today to X days in the future. Either this or a static date range is required. If a static date range is provided, days is ignored.
        date_from: If specifying a static date range, the from date, in "yyyy/mm/dd" format
        date_to: If specifying a static date range, the to date, in "yyyy/mm/dd" format
        sectors: Filter events by sectors. Values are "luxury", "designer", "high-street", "independent". Comma separated list. Leave blank for all sectors.
        types: Filter events by event type. Values are "brand-sales" ,"sample-sales", "vintage-sales", "in-store-events", "exhibitions", "parties-and-awards", "talks-and-plays", "trade-shows", "bridal-sales". Comma separated list. Leave blank for all types.
        designers: Filter events by designer, such as "chanel", "dior", "gucci". Check the designer URL on Chicmi.com for values. Comma separated list. Leave blank for all designers.
        stores: Filter events by a particular store, such as "200" = Harrods. Check the store URL on Chicmi for values. Comma separated list. Leave blank for all stores.
        max_results: The maximum number of results to return. Leaving this parameter off or providing 0 will return all results.
        
    """
    url = f"https://chicmi.p.rapidapi.com/calendar_in_city/"
    querystring = {'city': city, }
    if days:
        querystring['days'] = days
    if date_from:
        querystring['date_from'] = date_from
    if date_to:
        querystring['date_to'] = date_to
    if sectors:
        querystring['sectors'] = sectors
    if types:
        querystring['types'] = types
    if designers:
        querystring['designers'] = designers
    if stores:
        querystring['stores'] = stores
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chicmi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

