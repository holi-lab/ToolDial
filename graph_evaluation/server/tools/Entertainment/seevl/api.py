import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_related_entities(is_id: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related entities"
    id: Entity ID
        lang: Lang for results. Uses english if nothing is found for the given lang. Default: en
        
    """
    url = f"https://seevl.p.rapidapi.com/entities/{is_id}/related"
    querystring = {}
    if lang:
        querystring['_lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seevl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entity_search(preflabel: str, type: str=None, autocomplete: str=None, sort: str=None, page: str=None, maxres: str=None, lang: str=None, key: str='Entity key', value: str='Entity ID', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for an entity by prefLabel, i.e. its name. OR Search for an entity by key=value patterns. Values are entity IDs, and can be identified with the previous prefLabel search method."
    preflabel: Entity prefLabel
        type: Entity type
        autocomplete: Enable autocompletion. Default: 0
        sort: Sorting criteria. Defaut: alpha_asc
        page: Page. Default: 1
        maxres: Results per page. Default: 25, max: 100
        lang: Lang for matchin and results. Uses english if nothing is found for the given lang. Default: en
        key: string, required example: genre  Possible values: genre instrument label origin member.artist member.groupstring, required example: BntvuZAy
        value: string, required example: BntvuZAy
        
    """
    url = f"https://seevl.p.rapidapi.com/entities"
    querystring = {'prefLabel': preflabel, }
    if type:
        querystring['type'] = type
    if autocomplete:
        querystring['_autocomplete'] = autocomplete
    if sort:
        querystring['_sort'] = sort
    if page:
        querystring['_page'] = page
    if maxres:
        querystring['_maxres'] = maxres
    if lang:
        querystring['_lang'] = lang
    if key:
        querystring['key'] = key
    if value:
        querystring['value'] = value
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seevl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_basic_infos(is_id: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity basic infos (prefLabel, description, type). Description is from Wikipedia, you must display its source and licence when using it in your application."
    id: Entity ID
        lang: Lang for results. Uses english if nothing is found for the given lang. Default: en
        
    """
    url = f"https://seevl.p.rapidapi.com/entities/{is_id}/infos"
    querystring = {}
    if lang:
        querystring['_lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seevl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_fact_sheet(is_id: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity fact-sheet, including genre, label, influences, etc. for artists Fact-sheet is using a key-value model with keys including (depending on the entity type): * genre - musical genre (for Artist, SoloArtist, Group) * label - record label (for Artist, SoloArtist, Group) * origin - artist origin (for Artist, SoloArtist, Group) * membership.group - groups the artist is member of (for Artist, SoloArtist, Group) * membership.artist - group members (for Artist, SoloArtist, Group) * collaborated_with - artist collaborations (for Artist, SoloArtist, Group) * birth.place - birth place (for Artist, SoloArtist, Group) * birth.date - birth date (for Artist, SoloArtist, Group) * death.place - death place (for Artist, SoloArtist, Group) * death.date - death date (for Artist, SoloArtist, Group) * broader - paren genres (for Genre) * narrower - child genres (for Genre)"
    id: Entity ID
        lang: Lang for results. Uses english if nothing is found for the given lang. Default: en
        
    """
    url = f"https://seevl.p.rapidapi.com/entities/{is_id}/facts"
    querystring = {}
    if lang:
        querystring['_lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seevl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity_links(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entity links"
    id: Entity ID
        
    """
    url = f"https://seevl.p.rapidapi.com/entities/{is_id}/links"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seevl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_relations_between_entities(is_id: str, oid: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relations between entities. This method is only available for entities of type Artist (and its sub-types Solo and Group)"
    id: Entity ID
        oid: Entity ID
        lang: Lang for results. Uses english if nothing is found for the given lang. Default: en
        
    """
    url = f"https://seevl.p.rapidapi.com/entities/{is_id}/related/{oid}"
    querystring = {}
    if lang:
        querystring['_lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seevl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

