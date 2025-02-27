import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_many_cards(set: str='vivid-voltage', setid: str='33ee55f4-30d0-4900-850f-36a351fb9719', setnumber: str=None, cardnumber: str=None, name: str=None, limit: str=None, series: str='sword-and-shield', fromid: str=None, variant: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns many cards based on query parameters.
		
		A card represents a single card that can be defined by a combination of series / set / number in set / variant."
    limit: Max 200, defaults to 50
        fromid: A paging parameter. The ID from which the query will continue.
Calls for many results include a fromId field that can be used as a query param in the next call to get the next page of results
        variant: DEFAULT / REVERSE_HOLO
        
    """
    url = f"https://pokemon-tcg-card-prices.p.rapidapi.com/card"
    querystring = {}
    if set:
        querystring['set'] = set
    if setid:
        querystring['setId'] = setid
    if setnumber:
        querystring['setNumber'] = setnumber
    if cardnumber:
        querystring['cardNumber'] = cardnumber
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if series:
        querystring['series'] = series
    if fromid:
        querystring['fromId'] = fromid
    if variant:
        querystring['variant'] = variant
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-tcg-card-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_card_by_id(cardid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets details about a single card by it's ID."
    
    """
    url = f"https://pokemon-tcg-card-prices.p.rapidapi.com/card/{cardid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-tcg-card-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_set_by_id(setid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about a single set, retrieved using the ID for that set."
    
    """
    url = f"https://pokemon-tcg-card-prices.p.rapidapi.com/set/{setid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-tcg-card-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_many_sets(series: str='sword-and-shield', limit: int=20, fromid: str=None, set: str='vivid-voltage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns many sets based on query parameters.
		
		A set is a grouping of cards that are periodically released together. Every card has a set. Every set is part of a series."
    
    """
    url = f"https://pokemon-tcg-card-prices.p.rapidapi.com/set"
    querystring = {}
    if series:
        querystring['series'] = series
    if limit:
        querystring['limit'] = limit
    if fromid:
        querystring['fromId'] = fromid
    if set:
        querystring['set'] = set
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-tcg-card-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

