import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def collocations_v2_pos(lang: str, query: str, max_results: int=25, relation: str=None, min_sig: str=None, pos: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "looks up collocations
		new version in 2022: considers the part of speech of the query; some relation names changed"
    lang: language ISO 639-1 code for English
        query: any word (case insensitive)
        max_results: limit the number of results
        relation: One of the following relations (case sensitive): 
V:obj:N 
V:prep:N 
V:obj1+2:N 
V:obj+prep:N 
V:subj:N 
V:sc:Vinf (~~V:sc:V~~)
N:mod:Adj (~~N:mod:A~~)
N:prep:N 
N:nn:N 
V:mod:Adv (~~V:mod:A~~ )
Adj:mod:Adv (~~A:mod:A~~)

        min_sig: any number as the minimum significance (optional). Will return only collocations with significance >= min_sig
        pos: One of the following Part Of Speech (case sensitive): 
V
Vinf
N
Adj
Adv
        
    """
    url = f"https://linguatools-english-collocations.p.rapidapi.com/bolls/v2"
    querystring = {'lang': lang, 'query': query, }
    if max_results:
        querystring['max_results'] = max_results
    if relation:
        querystring['relation'] = relation
    if min_sig:
        querystring['min_sig'] = min_sig
    if pos:
        querystring['pos'] = pos
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linguatools-english-collocations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def collocations(lang: str, query: str, max_results: int=25, relation: str=None, min_sig: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "looks up collocations"
    lang: language ISO 639-1 code for English
        query: any word (case insensitive)
        max_results: limit the number of results
        relation: One of the following relations (case sensitive, optional): V:obj:N V:prep:N V:obj1+2:N V:obj+prep:N V:subj:N V:sc:V N:mod:A N:prep:N N:nn:N V:mod:A A:mod:A
        min_sig: any number as the minimum significance (optional). Will return only collocations with significance >= min_sig
        
    """
    url = f"https://linguatools-english-collocations.p.rapidapi.com/bolls/"
    querystring = {'lang': lang, 'query': query, }
    if max_results:
        querystring['max_results'] = max_results
    if relation:
        querystring['relation'] = relation
    if min_sig:
        querystring['min_sig'] = min_sig
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linguatools-english-collocations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

