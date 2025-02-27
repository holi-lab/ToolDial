import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def place_distance(placeid: str, distanceunit: str=None, toplaceid: str='Q60', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets this place's distance to the given place."
    distanceunit: The unit of distance: KM | MI [default]
        toplaceid: The distance to this place
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/places/{placeid}/distance"
    querystring = {}
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if toplaceid:
        querystring['toPlaceId'] = toplaceid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_time(placeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get this place's current time in ISO-8601 format: HHmmss.SSSZ"
    
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/places/{placeid}/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_date_time(placeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get this place's current date-time in ISO-6801 format: yyyyMMdd'T'HHmmssZ"
    
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/places/{placeid}/dateTime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_located_in(placeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for the containing populated place (e.g., its county or other administrative division), including location coordinates, population, and elevation above sea-level  (if available). 
		
		Currently, this data is highly dependent on whether the Wikidata **locatedIn** relation is properly defined. If you see an issue, please propose a change to the corresponding Wikidata entry."
    
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/places/{placeid}/locatedIn"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_details(placeid: str, asciimode: bool=None, languagecode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific place, including location coordinates, population, and elevation above sea-level (if available)."
    asciimode: Display results using ASCII characters
        languagecode: Display results in this language
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/places/{placeid}"
    querystring = {}
    if asciimode:
        querystring['asciiMode'] = asciimode
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_date_time(cityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the city current date-time in ISO-6801 format: yyyyMMdd'T'HHmmssZ"
    cityid: The city id (either native id or wikiDataId)
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/cities/{cityid}/dateTime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_located_in(cityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for the containing populated place (e.g., its county or other administrative division), including location coordinates, population, and elevation above sea-level  (if available). 
		
		Currently, this data is highly dependent on whether the Wikidata **locatedIn** relation is properly defined. If you see an issue, please propose a change to the corresponding Wikidata entry."
    
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/cities/{cityid}/locatedIn"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities(limit: int=None, location: str=None, radius: int=None, minpopulation: int=None, languagecode: str=None, asciimode: bool=None, distanceunit: str=None, timezoneids: str=None, sort: str=None, hateoasmode: bool=None, offset: int=None, types: str=None, countryids: str=None, excludedcountryids: str=None, includedeleted: str=None, nameprefixdefaultlangresults: bool=None, nameprefix: str=None, maxpopulation: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find cities, filtering by optional criteria. If no criteria are set, you will get back all known cities with a population of at least 1000."
    limit: The maximum number of results to retrieve
        location: Only cities near this location. Latitude/longitude in ISO-6709 format: ±DD.DDDD±DDD.DDDD
        radius: The location radius within which to find cities
        minpopulation: Only cities having at least this population
        languagecode: Display results in this language
        asciimode: Display results using ASCII characters
        distanceunit: The unit of distance to use: MI | KM
        timezoneids: Only cities in these time-zones
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        hateoasmode: Include HATEOAS-style links in results
        offset: The zero-ary offset into the results
        types: Only cities for these types (comma-delimited): CITY | ADM2
        countryids: Only cities in these countries (comma-delimited country codes or WikiData ids)
        excludedcountryids: Only cities NOT in these countries (comma-delimited country codes or WikiData ids)
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested.
        nameprefix: Only cities whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        maxpopulation: Only cities having no more than this population
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/cities"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if location:
        querystring['location'] = location
    if radius:
        querystring['radius'] = radius
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if languagecode:
        querystring['languageCode'] = languagecode
    if asciimode:
        querystring['asciiMode'] = asciimode
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if sort:
        querystring['sort'] = sort
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if offset:
        querystring['offset'] = offset
    if types:
        querystring['types'] = types
    if countryids:
        querystring['countryIds'] = countryids
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_near_location(radius: str, locationid: str, offset: int=None, limit: int=None, languagecode: str=None, timezoneids: str=None, asciimode: bool=None, includedeleted: str=None, maxpopulation: int=None, sort: str=None, hateoasmode: bool=None, excludedcountryids: str=None, nameprefixdefaultlangresults: bool=None, countryids: str=None, distanceunit: str=None, nameprefix: str=None, types: str=None, minpopulation: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get places near the given location, filtering by optional criteria."
    radius: The location radius within which to find places
        locationid: Only cities near this location. Latitude/longitude in ISO-6709 format: ±DD.DDDD±DDD.DDDD
        offset: The zero-ary offset into the results
        limit: The maximum number of results to retrieve
        languagecode: Display results in this language
        timezoneids: Only places in these time-zones
        asciimode: Display results using ASCII characters
        includedeleted: Whether to include any places marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        maxpopulation: Only places having no more than this population
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        hateoasmode: Include HATEOAS-style links in results
        excludedcountryids: Only places NOT in these countries (comma-delimited country codes or WikiData ids)
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        countryids: Only places in these countries (comma-delimited country codes or WikiData ids)
        distanceunit: The unit of distance to use: MI | KM
        nameprefix: Only places whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        types: Only places for these types (comma-delimited): ADM2 | CITY | ISLAND
        minpopulation: Only places having at least this population
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/locations/{locationid}/nearbyPlaces"
    querystring = {'radius': radius, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if languagecode:
        querystring['languageCode'] = languagecode
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if asciimode:
        querystring['asciiMode'] = asciimode
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if sort:
        querystring['sort'] = sort
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if countryids:
        querystring['countryIds'] = countryids
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if types:
        querystring['types'] = types
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_near_place(placeid: str, radius: int, asciimode: bool=None, languagecode: str=None, includedeleted: str=None, minpopulation: int=None, distanceunit: str=None, sort: str=None, hateoasmode: bool=None, nameprefixdefaultlangresults: bool=None, nameprefix: str=None, types: str=None, countryids: str=None, timezoneids: str=None, excludedcountryids: str=None, offset: int=None, maxpopulation: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get places near the given place, filtering by optional criteria."
    radius: The location radius within which to find places
        asciimode: Display results using ASCII characters
        languagecode: Display results in this language
        includedeleted: Whether to include any places marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        minpopulation: Only places having at least this population
        distanceunit: The unit of distance to use: MI | KM
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        hateoasmode: Include HATEOAS-style links in results
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        nameprefix: Only places whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        types: Only places for these types (comma-delimited): ADM2 | CITY | ISLAND
        countryids: Only places in these countries (comma-delimited country codes or WikiData ids)
        timezoneids: Only places in these time-zones
        excludedcountryids: Only places NOT in these countries (comma-delimited country codes or WikiData ids)
        offset: The zero-ary offset into the results
        maxpopulation: Only places having no more than this population
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/places/{placeid}/nearbyPlaces"
    querystring = {'radius': radius, }
    if asciimode:
        querystring['asciiMode'] = asciimode
    if languagecode:
        querystring['languageCode'] = languagecode
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if sort:
        querystring['sort'] = sort
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if types:
        querystring['types'] = types
    if countryids:
        querystring['countryIds'] = countryids
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if offset:
        querystring['offset'] = offset
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_places(countryid: str, limit: int=None, languagecode: str=None, hateoasmode: bool=None, asciimode: bool=None, offset: int=None, maxpopulation: int=None, timezoneids: str=None, types: str=None, sort: str=None, nameprefixdefaultlangresults: bool=None, includedeleted: str=None, minpopulation: int=None, nameprefix: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the places in the given country."
    limit: The maximum number of results to retrieve
        languagecode: Display results in this language
        hateoasmode: Include HATEOAS-style links in results
        asciimode: Display results using ASCII characters
        offset: The zero-ary offset into the results
        maxpopulation: Only places having no more than this population
        timezoneids: Only places in these time-zones
        types: Only cities for these types (comma-delimited): ADM2 | CITY | ISLAND
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD =  elevation | name | population
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        minpopulation: Only places having at least this population
        nameprefix: Only places whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries/{countryid}/places"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if languagecode:
        querystring['languageCode'] = languagecode
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if asciimode:
        querystring['asciiMode'] = asciimode
    if offset:
        querystring['offset'] = offset
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if types:
        querystring['types'] = types
    if sort:
        querystring['sort'] = sort
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_region_divisions(regioncode: str, countryid: str, limit: int=None, minpopulation: int=None, nameprefixdefaultlangresults: bool=None, languagecode: str=None, maxpopulation: int=None, sort: str=None, nameprefix: str=None, includedeleted: str=None, asciimode: bool=None, hateoasmode: bool=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the administrative divisions in the given region."
    regioncode: An ISO-3166 or FIPS region code
        countryid: An ISO-3166 country code or WikiData id
        limit: The maximum number of results to retrieve
        minpopulation: Only cities having at least this population
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        languagecode: Display results in this language
        maxpopulation: Only divisions having no more than this population
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD =  elevation | name | population
        nameprefix: Only divisions whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        asciimode: Display results using ASCII characters
        hateoasmode: Include HATEOAS-style links in results
        offset: The zero-ary offset into the results
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries/{countryid}/regions/{regioncode}/adminDivisions"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if languagecode:
        querystring['languageCode'] = languagecode
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if sort:
        querystring['sort'] = sort
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if asciimode:
        querystring['asciiMode'] = asciimode
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_region_cities(countryid: str, regioncode: str, limit: int=None, hateoasmode: bool=None, asciimode: bool=None, nameprefixdefaultlangresults: bool=None, timezoneids: str=None, nameprefix: str=None, types: str=None, minpopulation: int=None, languagecode: str=None, offset: int=None, maxpopulation: int=None, includedeleted: str=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the cities in the given region."
    countryid: An ISO-3166 country code or WikiData id
        regioncode: An ISO-3166 or FIPS region code
        limit: The maximum number of results to retrieve
        hateoasmode: Include HATEOAS-style links in results
        asciimode: Display results using ASCII characters
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        timezoneids: Only cities in these time-zones
        nameprefix: Only cities whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        types: Only cities for these types (comma-delimited): CITY | ADM2
        minpopulation: Only cities having at least this population
        languagecode: Display results in this language
        offset: The zero-ary offset into the results
        maxpopulation: Only cities having no more than this population
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD =  elevation | name | population
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries/{countryid}/regions/{regioncode}/cities"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if asciimode:
        querystring['asciiMode'] = asciimode
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if types:
        querystring['types'] = types
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if languagecode:
        querystring['languageCode'] = languagecode
    if offset:
        querystring['offset'] = offset
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_region_details(regioncode: str, countryid: str, asciimode: bool=None, languagecode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific country region, including number of cities."
    regioncode: An ISO-3166 or FIPS region code
        countryid: An ISO-3166 country code or WikiData id
        asciimode: Display results using ASCII characters
        languagecode: Display results in this language
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries/{countryid}/regions/{regioncode}"
    querystring = {}
    if asciimode:
        querystring['asciiMode'] = asciimode
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_regions(countryid: str, asciimode: bool=None, sort: str=None, languagecode: str=None, limit: int=None, hateoasmode: bool=None, offset: int=None, nameprefix: str=None, nameprefixdefaultlangresults: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find regions in a specific country, filtering by optional criteria. Regions can be states, provinces, districts, or otherwise major political divisions."
    countryid: An ISO-3166 country code or WikiData id
        asciimode: Display results using ASCII characters
        sort: How to sort the results. Format: ±SORT_FIELD where SORT_FIELD = fipsCode | isoCode | name
        languagecode: Display results in this language
        limit: The maximum number of results to retrieve
        hateoasmode: Include HATEOAS-style links in results
        offset: The zero-ary offset index into the results
        nameprefix: Only regions whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries/{countryid}/regions"
    querystring = {}
    if asciimode:
        querystring['asciiMode'] = asciimode
    if sort:
        querystring['sort'] = sort
    if languagecode:
        querystring['languageCode'] = languagecode
    if limit:
        querystring['limit'] = limit
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if offset:
        querystring['offset'] = offset
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_details(countryid: str, asciimode: bool=None, languagecode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific country, including number of regions."
    countryid: An ISO-3166 country code or WikiData id
        asciimode: Display results using ASCII characters
        languagecode: Display results in this language
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries/{countryid}"
    querystring = {}
    if asciimode:
        querystring['asciiMode'] = asciimode
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_time(cityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the city current time in ISO-8601 format: HHmmss.SSSZ"
    cityid: The city id (either native id or wikiDataId)
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/cities/{cityid}/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_distance(cityid: str, fromcityid: str=None, distanceunit: str=None, tocityid: str='Q60', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the distance to the given city."
    cityid: The city id (either native id or wikiDataId)
        fromcityid: The distance from this city
        distanceunit: The unit of distance: KM | MI [default]
        tocityid: The distance to this city
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/cities/{cityid}/distance"
    querystring = {}
    if fromcityid:
        querystring['fromCityId'] = fromcityid
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if tocityid:
        querystring['toCityId'] = tocityid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_near_location(locationid: str, radius: str, languagecode: str=None, maxpopulation: int=None, excludedcountryids: str=None, nameprefix: str=None, distanceunit: str=None, types: str=None, minpopulation: int=None, limit: int=None, asciimode: bool=None, offset: int=None, timezoneids: str=None, hateoasmode: bool=None, countryids: str=None, includedeleted: str=None, sort: str=None, nameprefixdefaultlangresults: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cities near the given location, filtering by optional criteria."
    locationid: Only cities near this location. Latitude/longitude in ISO-6709 format: ±DD.DDDD±DDD.DDDD
        radius: The location radius within which to find cities
        languagecode: Display results in this language
        maxpopulation: Only cities having no more than this population
        excludedcountryids: Only cities NOT in these countries (comma-delimited country codes or WikiData ids)
        nameprefix: Only cities whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        distanceunit: The unit of distance to use: MI | KM
        types: Only cities for these types (comma-delimited): CITY | ADM2
        minpopulation: Only cities having at least this population
        limit: The maximum number of results to retrieve
        asciimode: Display results using ASCII characters
        offset: The zero-ary offset into the results
        timezoneids: Only cities in these time-zones
        hateoasmode: Include HATEOAS-style links in results
        countryids: Only cities in these countries (comma-delimited country codes or WikiData ids)
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/locations/{locationid}/nearbyCities"
    querystring = {'radius': radius, }
    if languagecode:
        querystring['languageCode'] = languagecode
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if types:
        querystring['types'] = types
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if limit:
        querystring['limit'] = limit
    if asciimode:
        querystring['asciiMode'] = asciimode
    if offset:
        querystring['offset'] = offset
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if countryids:
        querystring['countryIds'] = countryids
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if sort:
        querystring['sort'] = sort
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def admin_division_details(divisionid: str, asciimode: bool=None, languagecode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific administrative division, including location coordinates, population, and elevation above sea-level (if available)."
    asciimode: Display results using ASCII characters
        languagecode: Display results in this language
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/adminDivisions/{divisionid}"
    querystring = {}
    if asciimode:
        querystring['asciiMode'] = asciimode
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_near_division(divisionid: str, radius: int, excludedcountryids: str=None, hateoasmode: bool=None, distanceunit: str=None, sort: str=None, maxpopulation: int=None, timezoneids: str=None, types: str=None, countryids: str=None, minpopulation: int=None, nameprefix: str=None, nameprefixdefaultlangresults: bool=None, limit: int=None, asciimode: bool=None, includedeleted: str=None, languagecode: str=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cities near the given administrative division, filtering by optional criteria."
    radius: The location radius within which to find cities
        excludedcountryids: Only cities NOT in these countries (comma-delimited country codes or WikiData ids)
        hateoasmode: Include HATEOAS-style links in results
        distanceunit: The unit of distance to use: MI | KM
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        maxpopulation: Only cities having no more than this population
        timezoneids: Only cities in these time-zones
        types: Only cities for these types (comma-delimited): CITY | ADM2
        countryids: Only cities in these countries (comma-delimited country codes or WikiData ids)
        minpopulation: Only cities having at least this population
        nameprefix: Only cities whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        limit: The maximum number of results to retrieve
        asciimode: Display results using ASCII characters
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        languagecode: Display results in this language
        offset: The zero-ary offset into the results
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/adminDivisions/{divisionid}/nearbyCities"
    querystring = {'radius': radius, }
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if sort:
        querystring['sort'] = sort
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if types:
        querystring['types'] = types
    if countryids:
        querystring['countryIds'] = countryids
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if limit:
        querystring['limit'] = limit
    if asciimode:
        querystring['asciiMode'] = asciimode
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if languagecode:
        querystring['languageCode'] = languagecode
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def admin_divisions(sort: str=None, excludedcountryids: str=None, radius: int=None, timezoneids: str=None, distanceunit: str=None, offset: int=None, location: str=None, nameprefix: str=None, maxpopulation: int=None, countryids: str=None, asciimode: bool=None, nameprefixdefaultlangresults: bool=None, minpopulation: int=None, includedeleted: str=None, limit: int=None, languagecode: str=None, hateoasmode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find administrative divisions, filtering by optional criteria. If no criteria are set, you will get back all known divisions with a population of at least 1000"
    sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        excludedcountryids: Only divisions NOT in these countries (comma-delimited country codes or WikiData ids)
        radius: The location radius within which to find divisions
        timezoneids: Only divisions in these time-zones
        distanceunit: The unit of distance to use: MI | KM
        offset: The zero-ary offset into the results
        location: Only divisions near this location. Latitude/longitude in ISO-6709 format: ±DD.DDDD±DDD.DDDD
        nameprefix: Only divisions whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        maxpopulation: Only divisions having no more than this population
        countryids: Only divisions in these countries (comma-delimited country codes or WikiData ids)
        asciimode: Display results using ASCII characters
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        minpopulation: Only divisions having at least this population
        includedeleted: Whether to include any divisions marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        limit: The maximum number of results to retrieve
        languagecode: Display results in this language
        hateoasmode: Include HATEOAS-style links in results
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/adminDivisions"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if radius:
        querystring['radius'] = radius
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if offset:
        querystring['offset'] = offset
    if location:
        querystring['location'] = location
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if countryids:
        querystring['countryIds'] = countryids
    if asciimode:
        querystring['asciiMode'] = asciimode
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if limit:
        querystring['limit'] = limit
    if languagecode:
        querystring['languageCode'] = languagecode
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_region_places(regionid: str, countryid: str, hateoasmode: bool=None, limit: int=None, languagecode: str=None, asciimode: bool=None, nameprefix: str=None, timezoneids: str=None, types: str=None, minpopulation: int=None, sort: str=None, maxpopulation: int=None, includedeleted: str=None, offset: int=None, nameprefixdefaultlangresults: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the places in the given region."
    countryid: An ISO-3166 country code or WikiData id
        hateoasmode: Include HATEOAS-style links in results
        limit: The maximum number of results to retrieve
        languagecode: Display results in this language
        asciimode: Display results using ASCII characters
        nameprefix: Only places whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        timezoneids: Only places in these time-zones
        types: Only cities for these types (comma-delimited): ADM2 | CITY | ISLAND
        minpopulation: Only places having at least this population
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD =  elevation | name | population
        maxpopulation: Only places having no more than this population
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        offset: The zero-ary offset into the results
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries/{countryid}/regions/{regionid}/places"
    querystring = {}
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if limit:
        querystring['limit'] = limit
    if languagecode:
        querystring['languageCode'] = languagecode
    if asciimode:
        querystring['asciiMode'] = asciimode
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if types:
        querystring['types'] = types
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if sort:
        querystring['sort'] = sort
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if offset:
        querystring['offset'] = offset
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places(limit: int=None, languagecode: str=None, timezoneids: str=None, asciimode: bool=None, offset: int=None, excludedcountryids: str=None, countryids: str=None, includedeleted: str=None, nameprefix: str=None, maxpopulation: int=None, distanceunit: str=None, sort: str=None, types: str=None, hateoasmode: bool=None, radius: int=None, minpopulation: int=None, nameprefixdefaultlangresults: bool=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find places, filtering by optional criteria. If no criteria are set, you will get back all known places."
    limit: The maximum number of results to retrieve
        languagecode: Display results in this language
        timezoneids: Only places in these time-zones
        asciimode: Display results using ASCII characters
        offset: The zero-ary offset into the results
        excludedcountryids: Only places NOT in these countries (comma-delimited country codes or WikiData ids)
        countryids: Only places in these countries (comma-delimited country codes or WikiData ids)
        includedeleted: Whether to include any places marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        nameprefix: Only places whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        maxpopulation: Only places having no more than this population
        distanceunit: The unit of distance to use: MI | KM
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        types: Only places for these types (comma-delimited): ADM2 | CITY | ISLAND
        hateoasmode: Include HATEOAS-style links in results
        radius: The location radius within which to find places
        minpopulation: Only places having at least this population
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested.
        location: Only places near this location. Latitude/longitude in ISO-6709 format: ±DD.DDDD±DDD.DDDD
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/places"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if languagecode:
        querystring['languageCode'] = languagecode
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if asciimode:
        querystring['asciiMode'] = asciimode
    if offset:
        querystring['offset'] = offset
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if countryids:
        querystring['countryIds'] = countryids
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if sort:
        querystring['sort'] = sort
    if types:
        querystring['types'] = types
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if radius:
        querystring['radius'] = radius
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(asciimode: bool=None, limit: int=None, hateoasmode: bool=None, offset: int=None, currencycode: str=None, languagecode: str=None, sort: str=None, nameprefixdefaultlangresults: bool=None, nameprefix: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find countries, filtering by optional criteria. If no criteria are set, you will get back all known countries."
    asciimode: Display results using ASCII characters
        limit: The maximum number of results to retrieve
        hateoasmode: Include HATEOAS-style links in results
        offset: The zero-ary offset index into the results
        currencycode: Only countries supporting this currency
        languagecode: Display results in this language
        sort: How to sort the results. Format: ±SORT_FIELD where SORT_FIELD = code | name
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        nameprefix: Only countries whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/countries"
    querystring = {}
    if asciimode:
        querystring['asciiMode'] = asciimode
    if limit:
        querystring['limit'] = limit
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if offset:
        querystring['offset'] = offset
    if currencycode:
        querystring['currencyCode'] = currencycode
    if languagecode:
        querystring['languageCode'] = languagecode
    if sort:
        querystring['sort'] = sort
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zone(zoneid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the time-zone current time in ISO-6801 format: HHmmss.SSSZ"
    zoneid: The time-zone id
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/locale/timezones/{zoneid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zones(offset: str=None, hateoasmode: bool=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all known time-zones."
    offset: The zero-ary offset index into the results
        hateoasmode: Include HATEOAS-style links in results
        limit: The maximum number of results to retrieve
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/locale/timezones"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(countryid: str='US', limit: int=None, hateoasmode: bool=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currencies, filtering by optional criteria. If no criteria are set, you will get back all known currencies."
    countryid: Only currencies supported by this country
        limit: The maximum number of results to retrieve
        hateoasmode: Include HATEOAS-style links in results
        offset: The zero-ary offset index into the results
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/locale/currencies"
    querystring = {}
    if countryid:
        querystring['countryId'] = countryid
    if limit:
        querystring['limit'] = limit
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_details(cityid: str, languagecode: str=None, asciimode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for a specific city, including location coordinates, population, and elevation above sea-level (if available)."
    cityid: The city id (either native id or wikiDataId)
        languagecode: Display results in this language
        asciimode: Display results using ASCII characters
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/cities/{cityid}"
    querystring = {}
    if languagecode:
        querystring['languageCode'] = languagecode
    if asciimode:
        querystring['asciiMode'] = asciimode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def admin_divisions_near_location(locationid: str, radius: str, excludedcountryids: str=None, offset: int=None, limit: int=None, languagecode: str=None, hateoasmode: bool=None, asciimode: bool=None, nameprefixdefaultlangresults: bool=None, countryids: str=None, minpopulation: int=None, includedeleted: str=None, maxpopulation: int=None, distanceunit: str=None, sort: str=None, timezoneids: str=None, nameprefix: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get administrative divisions near the given location, filtering by optional criteria."
    locationid: Only divisions near this location. Latitude/longitude in ISO-6709 format: ±DD.DDDD±DDD.DDDD
        radius: The location radius within which to find divisions
        excludedcountryids: Only divisions NOT in these countries (comma-delimited country codes or WikiData ids)
        offset: The zero-ary offset into the results
        limit: The maximum number of results to retrieve
        languagecode: Display results in this language
        hateoasmode: Include HATEOAS-style links in results
        asciimode: Display results using ASCII characters
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        countryids: Only divisions in these countries (comma-delimited country codes or WikiData ids)
        minpopulation: Only divisions having at least this population
        includedeleted: Whether to include any divisions marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        maxpopulation: Only divisions having no more than this population
        distanceunit: The unit of distance to use: MI | KM
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        timezoneids: Only divisions in these time-zones
        nameprefix: Only divisions whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/locations/{locationid}/nearbyDivisions"
    querystring = {'radius': radius, }
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if languagecode:
        querystring['languageCode'] = languagecode
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if asciimode:
        querystring['asciiMode'] = asciimode
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if countryids:
        querystring['countryIds'] = countryids
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if sort:
        querystring['sort'] = sort
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def admin_divisions_near_division(divisionid: str, radius: int, minpopulation: int=None, hateoasmode: bool=None, nameprefixdefaultlangresults: bool=None, maxpopulation: int=None, distanceunit: str=None, asciimode: bool=None, offset: int=None, excludedcountryids: str=None, languagecode: str=None, limit: int=None, countryids: str=None, nameprefix: str=None, timezoneids: str=None, sort: str=None, includedeleted: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get divisions near the given administrative division, filtering by optional criteria."
    radius: The location radius within which to find divisions
        minpopulation: Only divisions having at least this population
        hateoasmode: Include HATEOAS-style links in results
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        maxpopulation: Only divisions having no more than this population
        distanceunit: The unit of distance to use: MI | KM
        asciimode: Display results using ASCII characters
        offset: The zero-ary offset into the results
        excludedcountryids: Only divisions NOT in these countries (comma-delimited country codes or WikiData ids)
        languagecode: Display results in this language
        limit: The maximum number of results to retrieve
        countryids: Only divisions in these countries (comma-delimited country codes or WikiData ids)
        nameprefix: Only divisions whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        timezoneids: Only divisions in these time-zones
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        includedeleted: Whether to include any divisions marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/adminDivisions/{divisionid}/nearbyDivisions"
    querystring = {'radius': radius, }
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if asciimode:
        querystring['asciiMode'] = asciimode
    if offset:
        querystring['offset'] = offset
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if languagecode:
        querystring['languageCode'] = languagecode
    if limit:
        querystring['limit'] = limit
    if countryids:
        querystring['countryIds'] = countryids
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if sort:
        querystring['sort'] = sort
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zone_time(zoneid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the time-zone current time in ISO-6801 format: HHmmss.SSSZ"
    zoneid: The time-zone id
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/locale/timezones/{zoneid}/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zone_date_time(zoneid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the timezone current date-time in ISO-6801 format: yyyyMMdd'T'HHmmssZ"
    zoneid: The time-zone id
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/locale/timezones/{zoneid}/dateTime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(hateoasmode: bool=None, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all supported languages"
    hateoasmode: Include HATEOAS-style links in results
        offset: The zero-ary offset index into the results
        limit: The maximum number of results to retrieve
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/locale/languages"
    querystring = {}
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locales(hateoasmode: bool=None, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all known locales."
    hateoasmode: Include HATEOAS-style links in results
        limit: The maximum number of results to retrieve
        offset: The zero-ary offset index into the results
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/locale/locales"
    querystring = {}
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_near_city(cityid: str, radius: int, limit: int=None, offset: int=None, timezoneids: str=None, minpopulation: int=None, sort: str=None, languagecode: str=None, asciimode: bool=None, maxpopulation: int=None, nameprefix: str=None, nameprefixdefaultlangresults: bool=None, types: str=None, distanceunit: str=None, hateoasmode: bool=None, countryids: str=None, excludedcountryids: str=None, includedeleted: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cities near the given city, filtering by optional criteria."
    cityid: The city id (either native id or wikiDataId)
        radius: The location radius within which to find cities
        limit: The maximum number of results to retrieve
        offset: The zero-ary offset into the results
        timezoneids: Only cities in these time-zones
        minpopulation: Only cities having at least this population
        sort: How to sort the results. Format: ±SORT_FIELD,±SORT_FIELD where SORT_FIELD = countryCode | elevation | name | population
        languagecode: Display results in this language
        asciimode: Display results using ASCII characters
        maxpopulation: Only cities having no more than this population
        nameprefix: Only cities whose names start with this prefix. If languageCode is set, the prefix will be matched on the name as it appears in that language.
        nameprefixdefaultlangresults: When name-prefix matching, whether or not to match on names in the default language if a non-default language is requested
        types: Only cities for these types (comma-delimited): CITY | ADM2
        distanceunit: The unit of distance to use: MI | KM
        hateoasmode: Include HATEOAS-style links in results
        countryids: Only cities in these countries (comma-delimited country codes or WikiData ids)
        excludedcountryids: Only cities NOT in these countries (comma-delimited country codes or WikiData ids)
        includedeleted: Whether to include any cities marked deleted: ALL | SINCE_YESTERDAY | SINCE_LAST_WEEK | NONE
        
    """
    url = f"https://wft-geo-db.p.rapidapi.com/v1/geo/cities/{cityid}/nearbyCities"
    querystring = {'radius': radius, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if timezoneids:
        querystring['timeZoneIds'] = timezoneids
    if minpopulation:
        querystring['minPopulation'] = minpopulation
    if sort:
        querystring['sort'] = sort
    if languagecode:
        querystring['languageCode'] = languagecode
    if asciimode:
        querystring['asciiMode'] = asciimode
    if maxpopulation:
        querystring['maxPopulation'] = maxpopulation
    if nameprefix:
        querystring['namePrefix'] = nameprefix
    if nameprefixdefaultlangresults:
        querystring['namePrefixDefaultLangResults'] = nameprefixdefaultlangresults
    if types:
        querystring['types'] = types
    if distanceunit:
        querystring['distanceUnit'] = distanceunit
    if hateoasmode:
        querystring['hateoasMode'] = hateoasmode
    if countryids:
        querystring['countryIds'] = countryids
    if excludedcountryids:
        querystring['excludedCountryIds'] = excludedcountryids
    if includedeleted:
        querystring['includeDeleted'] = includedeleted
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

