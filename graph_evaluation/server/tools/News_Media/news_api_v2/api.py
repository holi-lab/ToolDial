import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_headlines(language: str='en', category: str='sports', country: str='us', sortby: str=None, pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to find top headlines news."
    language: en, zh-Hans, zh-Hant, id, cs, uk, he, ar, de, es-419, ja, ko, fr, it, lv, lt, ml, th, hu, nl, no, pl, pt-419, pt-150, ro, sk, sl, sv, vi, tr, el, bg, ru, sr, mr, hi, bn, ta, te
        category: Possible options: business, science, sports, entertainment, health, technology

Default value: all categories
        country: MY, GB, CN, TW, AU, BW, ET, KR, GH, IE, KE, LV, NA, IN, BD, TH, NZ, NG, PK, PH, SG, ZA, TZ, UG, ZW, ID, CZ, DE, AT, CH, AR, EG, CL, CO, CU, US, MX, PE, VE, LB, CA, FR, MA, SN, IT, LT, HK, JP, HU, BE, NL, NO, PL, BR, PT, RO, SK, SI, SE, VN, TR, GR, BG, RU, RS, UA, IL, AE, SA
        sortby: Default value: published_date
        
    """
    url = f"https://news-api14.p.rapidapi.com/top-headlines"
    querystring = {}
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    if country:
        querystring['country'] = country
    if sortby:
        querystring['sortBy'] = sortby
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-api14.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, to: str=None, pagesize: int=10, is_from: str=None, country: str='us', language: str='en', sortby: str=None, publisher: str='cnn.com,bbc.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to search news by specific keywords."
    to: The datetime string must match ISO 8601 format
        is_from: The datetime string must match ISO 8601 format
        country: MY, GB, CN, TW, AU, BW, ET, KR, GH, IE, KE, LV, NA, IN, BD, TH, NZ, NG, PK, PH, SG, ZA, TZ, UG, ZW, ID, CZ, DE, AT, CH, AR, EG, CL, CO, CU, US, MX, PE, VE, LB, CA, FR, MA, SN, IT, LT, HK, JP, HU, BE, NL, NO, PL, BR, PT, RO, SK, SI, SE, VN, TR, GR, BG, RU, RS, UA, IL, AE, SA
        language: en, zh-Hans, zh-Hant, id, cs, uk, he, ar, de, es-419, ja, ko, fr, it, lv, lt, ml, th, hu, nl, no, pl, pt-419, pt-150, ro, sk, sl, sv, vi, tr, el, bg, ru, sr, mr, hi, bn, ta, te
        sortby: Default value: random
        publisher: A Comma-separated string of publisher's.

Maximum 5 publishers.
        
    """
    url = f"https://news-api14.p.rapidapi.com/search"
    querystring = {'q': q, }
    if to:
        querystring['to'] = to
    if pagesize:
        querystring['pageSize'] = pagesize
    if is_from:
        querystring['from'] = is_from
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    if sortby:
        querystring['sortBy'] = sortby
    if publisher:
        querystring['publisher'] = publisher
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-api14.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

