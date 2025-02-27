import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_subregions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of sub regions in the world"
    
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/subregion"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sub_regions(region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of sub regions of a region"
    region: Name of the region
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/subregion/{region}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_language(language: str, offset: int=0, limit: int=10, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by its used Language"
    language: The language used within the country. User can provide both language code or language name for searching. This search is case insensitve and exact match
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        limit: The maximum records that can be returned within a page
        fields: Comma separated fields list to filter response
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/language/{language}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_timezone(timezone: str, limit: int=10, fields: str=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries where provided TimeZone is used by the country"
    timezone: TimeZone in format (UTC+/-HH:MM). Example: UTC-04:00 or UTC+01:00
        limit: The maximum records that can be returned within a page
        fields: Comma separated fields list to filter response
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/timeZone/{timezone}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_region(region: str, offset: int=0, fields: str=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by its Region"
    region: The region of the country. This search is case insensitive and exact match
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        fields: Comma separated fields list to filter response
        limit: The maximum records that can be returned within a page
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/region/{region}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries(fields: str=None, status: str='officially-assigned', landlocked: bool=None, subregion: str=None, startofweek: str='Monday', independent: bool=None, unmember: bool=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief details of all Countries"
    fields: Comma separated fields list to filter response
        status: Status of the country.
        landlocked: Is the country landlocked. Supported values are true/false
        subregion: Sub-Region the country belongs to, like Caribbean
        startofweek: Country by start day of the week
        independent: Countries who are recognized as independent. Values true/false.
        unmember: Is the country member of UN. Supported values are true/false
        region: Region the country belongs to, like Americas
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if status:
        querystring['status'] = status
    if landlocked:
        querystring['landlocked'] = landlocked
    if subregion:
        querystring['subregion'] = subregion
    if startofweek:
        querystring['startOfWeek'] = startofweek
    if independent:
        querystring['independent'] = independent
    if unmember:
        querystring['unMember'] = unmember
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_sub_region(subregion: str, fields: str=None, limit: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by its Sub-Region"
    subregion: The subregion of the country. This search is case insensitive and exact match
        fields: Comma separated fields list to filter response
        limit: The maximum records that can be returned within a page
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/subregion/{subregion}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_continents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of continents in the world"
    
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/continent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_capital(capital: str, limit: int=10, offset: int=0, fields: str=None, exactmatch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by capital. Both official and common names are searched"
    capital: Capital of the Country to be searched. By default like query type search is supported where a country can be search by only providing part of the capital name. If exact match is needed, add exactMatch=true in query parameter.
        limit: The maximum records that can be returned within a page
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        fields: Comma separated fields list to filter response
        exactmatch: Set to true if exact name of the capital match is required. Default value is false
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/capital/{capital}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if fields:
        querystring['fields'] = fields
    if exactmatch:
        querystring['exactMatch'] = exactmatch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_idd(idd: str, limit: int=10, offset: int=0, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by its International Direct Dialing Number"
    idd: Idd number starting with +
        limit: The maximum records that can be returned within a page
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        fields: Comma separated fields list to filter response
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/idd/{idd}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_neighbour_countries_by_country_code(countrycode: str, offset: int=0, limit: int=10, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search broder sharing countries for given country code. Country code is ISO 3166 standard alpha code"
    countrycode: ISO 3166 standard alpha code of the country
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        limit: The maximum records that can be returned within a page
        fields: Comma separated fields list to filter response
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/neighbour/{countrycode}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_continent(continent: str, offset: int=0, fields: str=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by its Continent"
    continent: The continent of the country. This search is case insensitive and exact match
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        fields: Comma separated fields list to filter response
        limit: The maximum records that can be returned within a page
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/continent/{continent}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_population(fields: str=None, offset: int=0, minpopulation: int=None, limit: int=10, maxpopulation: int=None, sortorder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search countries by minimum population or maximum population or between minimum and maximum population. Search operation will include both upper and lower limit."
    fields: Comma separated fields list to filter response
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        minpopulation: Minimum population threshold
        limit: The maximum records that can be returned within a page
        maxpopulation: Maximum population threshold
        sortorder: Sort countries by population. Default sortOrder=asc.
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/population"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if offset:
        querystring['offset'] = offset
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if limit:
        querystring['limit'] = limit
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if sortorder:
        querystring['sortOrder'] = sortorder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_name(name: str, offset: int=0, exactmatch: bool=None, limit: int=10, includenativename: bool=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by Name. Both official and common names are searched"
    name: Name of the Country to be searched. By default like query type search is supported where a country can be search by only providing part of the name. If exact match is needed, add exactMatch=true in query parameter.
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        exactmatch: Set to true if exact name match is required. Default value is false
        limit: The maximum records that can be returned within a page
        includenativename: If the name search should include Native Name as well, set it to true
        fields: Comma separated fields list to filter response
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/name/{name}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if exactmatch:
        querystring['exactMatch'] = exactmatch
    if limit:
        querystring['limit'] = limit
    if includenativename:
        querystring['includeNativeName'] = includenativename
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries_by_currency(currency: str, limit: int=10, fields: str=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Countries by currency used by the country. The search can be based on currency code or currency name. By default substring match is used. If exact match is required, one should pass query parameter "exactMatch=true""
    currency: The Currency used within the country. User can provide both currency code or currency name for searching. This search is case insensitive
        limit: The maximum records that can be returned within a page
        fields: Comma separated fields list to filter response
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/currency/{currency}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_country_by_fifa_code(fifacode: str, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Country by FIFA assigned Country Code"
    fifacode: FIFA assigned Country Code
        fields: Comma separated fields list to filter response
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/fifa/{fifacode}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_country_by_alphacode(code: str, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief countries details by ISO 3166 standard alpha code"
    code: ISO 3166 standard alpha code of the country
        fields: Comma separated fields list to filter response
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/country/{code}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of regions in the world"
    
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/region"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities(countrycode: str, fields: str=None, limit: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details about cities"
    countrycode: Alpha-2 code of Country as per ISO-3166
        fields: Comma separated fields list to filter response
        limit: The maximum records that can be returned within a page
        offset: The page number from which the search should continue. Its zero indexed, hence, starts with 0
        
    """
    url = f"https://geography4.p.rapidapi.com/apis/geography/v1/city"
    querystring = {'countryCode': countrycode, }
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geography4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

