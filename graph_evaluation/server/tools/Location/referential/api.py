import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def city_by_ip_address_geo_ip(ip: str, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a city, state, country and continent by IP address. For some results, the closest result within 10 miles will be provided."
    ip: ip4 pr ip6 formatted IP address
        fields: Select fields to return in result [geonameid, timezone,state_code,iso_a2]
        
    """
    url = f"https://referential.p.rapidapi.com/v1/city"
    querystring = {'ip': ip, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(currency: str=None, dial_code: int=None, currency_code: str=None, fields: str='currency,currency_num_code,currency_code,continent_code,currency,iso_a3,dial_code', iso_a2: str=None, name: str=None, limit: int=250, continent_code: str=None, iso_a3: str=None, currency_num_code: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List or search countries of the world in different languages.. Get iso a2, a3 country code and localized country name"
    currency: Filter on currency name in chosen lang
        dial_code: Filter on country dial code
        currency_code: Filter on 3 letter currency code (ISO)
        fields: Select fields to return for each country [iso_a2, iso_a3, continent_code, dial_code, currency_code, currency_num_code, currency]
        iso_a2: Filter on 2 letter iso country code
        name: Country name search string. Can be partial contains...
        continent_code: Two letter region/continent code (eu, na, sa, as, af, oc, an). Lookup at /v1/continent.
        iso_a3: Filter on 3 letter iso country code
        currency_num_code: Filter on 3 digit numeric currency code (ISO)
        lang: Language selection ISO 639-2 letter code. Overrides Accept-Language header. Defaults to browser value. Default: en
        
    """
    url = f"https://referential.p.rapidapi.com/v1/country"
    querystring = {}
    if currency:
        querystring['currency'] = currency
    if dial_code:
        querystring['dial_code'] = dial_code
    if currency_code:
        querystring['currency_code'] = currency_code
    if fields:
        querystring['fields'] = fields
    if iso_a2:
        querystring['iso_a2'] = iso_a2
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if continent_code:
        querystring['continent_code'] = continent_code
    if iso_a3:
        querystring['iso_a3'] = iso_a3
    if currency_num_code:
        querystring['currency_num_code'] = currency_num_code
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities(limit: int=250, state_hasc: str=None, timezone: str=None, geonameid: str=None, prefix: str='san fr', timezone_offset: int=None, sort: str=None, lang: str='en', iso_a2: str='us', name: str=None, state_code: str='US-CA', fields: str='iso_a2,state_code,state_hasc,timezone,timezone_offset', order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List and search over 75,000 world cities in 20 languages."
    limit: Limit results to N results
        state_hasc: Filter on state's HASC code. List at /v1/state.
        timezone: Filter on timezone name. List at /v1/timezone.
        geonameid: Filter on geonameid
        prefix: Search by names starting with prefix. See also limit, order and sort fields.
        timezone_offset: Filter on timezone offset in hours
        sort: Sort on field [population, state_code, name, timezone, state_hasc, iso_a2]. Please see limit and order fields.
        lang: Language selection ISO 639-2 letter code. Overrides Accept-Language header. Defaults to browser value. Default: en
        iso_a2: Filter on 2 letter country code
        name: Search as \"contains\" on city name in chosen lang
        state_code: Filter on state's ISO-3166-2 letter code. List at /v1/state
        fields: Select fields to return in result [iso_a2,state_code,timezone,timezone_offset,state_hasc,geonameid]
        order: Sort order [asc | desc]
        
    """
    url = f"https://referential.p.rapidapi.com/v1/city"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if state_hasc:
        querystring['state_hasc'] = state_hasc
    if timezone:
        querystring['timezone'] = timezone
    if geonameid:
        querystring['geonameid'] = geonameid
    if prefix:
        querystring['prefix'] = prefix
    if timezone_offset:
        querystring['timezone_offset'] = timezone_offset
    if sort:
        querystring['sort'] = sort
    if lang:
        querystring['lang'] = lang
    if iso_a2:
        querystring['iso_a2'] = iso_a2
    if name:
        querystring['name'] = name
    if state_code:
        querystring['state_code'] = state_code
    if fields:
        querystring['fields'] = fields
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state(limit: int=250, iso_a2: str='us', iso_3166_2: str=None, fields: str='iso_a2', name: str='tex', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the sub-regions/states/provinces/departments of the world's countries"
    iso_a2: Filter on 2 letter country code
        iso_3166_2: Filter on Subregion's ISO-3166-2 letter code
        fields: Select fields to select in the result [iso_a2]
        name: Filter as \"contains\" on the states's name in chosen lang
        lang: Language selection ISO 639-2 letter code. Overrides Accept-Language header. Defaults to browser value. Default: en
        
    """
    url = f"https://referential.p.rapidapi.com/v1/state"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if iso_a2:
        querystring['iso_a2'] = iso_a2
    if iso_3166_2:
        querystring['iso_3166_2'] = iso_3166_2
    if fields:
        querystring['fields'] = fields
    if name:
        querystring['name'] = name
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezones(offset: str=None, name: str=None, code: str=None, timezone: str=None, daylights_offset: str=None, lang: str='de', daylights_code: str=None, fields: str='offset,daylights_offset,daylights,daylights_code,timezone', daylights: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the world's time zones in 20 languages"
    offset: Filter on non daylights savings offset in hours
        name: Filter as "contains" on timezone name in chosen lang
        code: Filter on 3 letter non daylight savings timezone code (not unique)
        timezone: Filter on timezone name in english
        daylights_offset: Filter on daylights savings offset in hours
        lang: Language selection ISO 639-2 letter code. Overrides Accept-Language header. Defaults to browser value. Default: en
        daylights_code: Filter on daylights savings letter code (not unique)
        fields: Select fields to return in result [code,timezone,daylights_code,daylights_offset,daylights,hours_offset,offset]
        daylights: Filter on daylights savings/non daylight savings timezones (true,false)
        
    """
    url = f"https://referential.p.rapidapi.com/v1/timezone"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    if code:
        querystring['code'] = code
    if timezone:
        querystring['timezone'] = timezone
    if daylights_offset:
        querystring['daylights_offset'] = daylights_offset
    if lang:
        querystring['lang'] = lang
    if daylights_code:
        querystring['daylights_code'] = daylights_code
    if fields:
        querystring['fields'] = fields
    if daylights:
        querystring['daylights'] = daylights
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_by_id(is_id: str, lang: str='en', fields: str='iso_a2,state_code,state_hasc,timezone, timezone_offset', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Direct lookup of over 75,000 cities by id. See the v1/city API to search for city ids."
    id: Lookup at /v1/city.
        lang: Language selection ISO 639-2 letter code. Overrides Accept-Language header. Defaults to browser value. Default: en
        fields: Select fields to return in result [iso_a2,state_code,timezone,timezone_offset,state_hasc]
        
    """
    url = f"https://referential.p.rapidapi.com/v1/city/{is_id}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_by_iso_code(iso_code: str, fields: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get country by 2 letter iso code localized to a given language"
    iso_code: 2 letter iso country code. 
        fields: Comma deliminated list of fields to return in response
        lang: 2 letter iso language code to localize country name
        
    """
    url = f"https://referential.p.rapidapi.com/v1/country/{iso_code}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_by_id(code: str, fields: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup sub-regions/states/provinces/departments by iso 3166-2 code"
    code: ISO 3166-2 state/province/deparm
        fields: Select fields to return in result [iso_a2]
        lang: Language selection ISO 639-2 letter code. Overrides Accept-Language header. Defaults to browser value. Default: en
        
    """
    url = f"https://referential.p.rapidapi.com/v1/state/{code}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def continent_by_id(code: str, lang: str=None, continent_code: str=None, name: str=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup a continent by id"
    code: Two letter iso continent code. Lookup /v1/continent.
        lang: Two letter iso 639-2 language code (i.e. en, sv, pl etc). Lookup /v1/lang.
        continent_code: Filter on 2 letter continent code
        name: Filter on continent name (value field) as \"contains\"
        fields: Select fields to return for each continent [iso_a2, iso_a3, continent_code, dial_code, currency_code, currency_num_code, currency]
        
    """
    url = f"https://referential.p.rapidapi.com/v1/continent/{code}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if continent_code:
        querystring['continent_code'] = continent_code
    if name:
        querystring['name'] = name
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def continents(fields: str=None, continent_code: str=None, lang: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List continents of the world localized to requested language"
    fields: Select fields to return for each continent [iso_a2, iso_a3, continent_code, dial_code, currency_code, currency_num_code, currency]
        continent_code: Filter on 2 letter continent code
        lang: Two letter iso 639-2 language code (i.e. en, sv, pl etc). Lookup /v1/lang.
        name: Filter as "contains" on continent name in chosen lang
        
    """
    url = f"https://referential.p.rapidapi.com/v1/continent"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if continent_code:
        querystring['continent_code'] = continent_code
    if lang:
        querystring['lang'] = lang
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(lang_3: str=None, iso_a2: str=None, fields: str='iso_a2,lang_3,flag', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the world's languages translated into 20 languages"
    lang_3: Filter by 3 letter ISO language code
        iso_a2: Filter on 2 letter country code
        fields: Select fields to return in result [flag,iso_a2,lang_3,lang]
        lang: Returns a language translated into lang language
        
    """
    url = f"https://referential.p.rapidapi.com/v1/lang"
    querystring = {}
    if lang_3:
        querystring['lang_3'] = lang_3
    if iso_a2:
        querystring['iso_a2'] = iso_a2
    if fields:
        querystring['fields'] = fields
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language_by_id(lang: str, lang_3: str=None, iso_a2: str=None, fields: str='iso_a2,lang_3,flag', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find language by id translated into 20 languages"
    lang: 2 letter iso language code. Lookup at /v1/lang.
        lang: Returns a language translated into lang language
        lang_3: Filter by 3 letter ISO language code
        iso_a2: Filter on 2 letter country code
        fields: Select fields to return in result [flag,iso_a2,lang_3,lang]
        
    """
    url = f"https://referential.p.rapidapi.com/v1/lang/{lang}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if lang_3:
        querystring['lang_3'] = lang_3
    if iso_a2:
        querystring['iso_a2'] = iso_a2
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezone_by_id(timezone: str, fields: str='offset,daylights_offset,daylights,daylights_code,timezone', lang: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup a time zone"
    timezone: Timezone
        fields: Select fields to return in result [code,timezone,daylights_code,daylights_offset,daylights,hours_offset,offset]
        lang: Language selection ISO 639-2 letter code. Overrides Accept-Language header. Defaults to browser value. Default: en
        
    """
    url = f"https://referential.p.rapidapi.com/v1/timezone/{timezone}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "referential.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

