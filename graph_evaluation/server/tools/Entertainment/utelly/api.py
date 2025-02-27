import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lookup(country: str='uk', term: str='bojack', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup a tv show or movie by name and retrieve its availability across supported services for a particular country || Netflix, Amazon Prime Video, Amazon Instant Video, Apple TV+, Google Play, iTunes, YouTube Premium, Disney Plus, Hulu, Atom Tickets, CBS, DC Universe, HBO, Discovery Channel, Fandango Movies, Fox, NBC, Nickelodeon."
    country: only selects services available in a specific country - supported territories uk, us, ar, at, be, br, ca, de, es, fr, ie, id, it, is, kr, my, mx, no, nl, pt,  se & sg
        term: name of the show you want availability - supports partial strings
        
    """
    url = f"https://utelly-tv-shows-and-movies-availability-v1.p.rapidapi.com/lookup"
    querystring = {}
    if country:
        querystring['country'] = country
    if term:
        querystring['term'] = term
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "utelly-tv-shows-and-movies-availability-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def idlookup(source_id: str, source: str, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find where to watch a movie or tv show, with look-up using common IDs (IMDb, tmdb, wiki, etc). 
		
		For example (The Shawshank Redemption): 
		
		- **IMDb**, tt0111161
		-  **TMDb**, movie/278"
    country: only selects services available in a specific country - supported territories uk, us, ar, at, be, br, ca, de, es, fr, ie, id, it, is, kr, my, mx, no, nl, pt,  se & sg
        
    """
    url = f"https://utelly-tv-shows-and-movies-availability-v1.p.rapidapi.com/idlookup"
    querystring = {'source_id': source_id, 'source': source, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "utelly-tv-shows-and-movies-availability-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

