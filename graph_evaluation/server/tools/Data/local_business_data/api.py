import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def business_details(business_id: str, extract_share_link: bool=None, experimental: bool=None, region: str='us', extract_emails_and_contacts: bool=None, fields: str=None, language: str='en', coordinates: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full business details including emails and social contacts. Supports batching of up to 20 Business Ids."
    business_id: Unique Business Id. Accepts google_id / business_id and google_place_id. Examples of valid ids: `0x880fd393d427a591:0x8cba02d713a995ed`, `ChIJkaUn1JPTD4gR7ZWpE9cCuow`.

In addition, batching of up to 20 Business Ids is supported in a single request using a comma separated list (e.g. business_id=id1,id2).
        extract_share_link: Whether to extract place's share link for the business. In case true, businesses will be enriched with a `share_link` field containing a shortened Google URL for sharing (e.g. https://goo.gl/maps/oxndE8SVaNU5CV6p6).

**Default**: *`false`*
        experimental: Experimental flag
        region: Query Google Maps from a particular region or country. For a list of supported region/country codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (Alpha-2 code).

**Default:** *`us`*
        extract_emails_and_contacts: Whether to extract emails, contacts and social profiles for the business. In case true, businesses will be enriched with a `emails_and_contacts` field, potentially containing emails, phones, Facebook, LinkedIn, Instagram and other social profiles.

**Default**: *`true`*
        fields: A comma separated list of business fields to include in the response (field projection). By default all fields are returned.

**e.g.** `business_id,type,phone_number,full_address`
        language: Set the language of the results. For a list of supported language codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes .

**Default**: *`en`*
        coordinates: Geographic coordinates of the location from which the query is applied - recommended to use so that results are biased towards this location. Defaults to some central location in the region (see the `region` parameter).
        
    """
    url = f"https://local-business-data.p.rapidapi.com/business-details"
    querystring = {'business_id': business_id, }
    if extract_share_link:
        querystring['extract_share_link'] = extract_share_link
    if experimental:
        querystring['experimental'] = experimental
    if region:
        querystring['region'] = region
    if extract_emails_and_contacts:
        querystring['extract_emails_and_contacts'] = extract_emails_and_contacts
    if fields:
        querystring['fields'] = fields
    if language:
        querystring['language'] = language
    if coordinates:
        querystring['coordinates'] = coordinates
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, x_user_agent: str=None, region: str='us', business_status: str=None, subtypes: str=None, verified: bool=None, lat: str='37.359428', limit: str='20', language: str='en', zoom: str='13', fields: str=None, lng: str='-121.925337', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search local businesses on Google Maps."
    query: Search query / keyword.

**e.g.** `Plumbers near New-York, USA`
**e.g.** `Bars in 94102, USA`
        x_user_agent: Device type for the search. Default `desktop`.
        region: Query Google Maps from a particular region or country. For a list of supported region/country codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (Alpha-2 code).

**Default:** *`us`*
        business_status: Find businesses with specific status, specified as a comma separated list of the following values: `OPEN`, `CLOSED_TEMPORARILY`, `CLOSED`.

**e.g.** `OPEN`
**e.g.** `CLOSED_TEMPORARILY,CLOSED`
        subtypes: Find businesses with specific subtypes, specified as a comma separated list of types (business categories). For the complete list of types, see https://daltonluka.com/blog/google-my-business-categories.

**e.g.** `Plumber,Carpenter,Electrician`
**e.g.** `Night club,Dance club,Bar,Pub`
        verified: Only return verified businesses
        lat: Latitude of the coordinates point from which the query is applied - recommended to use so that results are biased towards this location. Defaults to some central location in the region (see the `region` parameter).
        limit: Maximum number of businesses to return (1-500).

**Default**: *`20`*
        language: Set the language of the results. For a list of supported language codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes .

**Default**: *`en`*
        zoom: Zoom level on which to make the search (the viewport is determined by lat, lng and zoom).

**Default**: *`13`*
        fields: A comma separated list of business fields to include in the response (field projection). By default all fields are returned.

**e.g.** `business_id,type,phone_number,full_address`
        lng: Longitude of the coordinates point from which the query is applied - recommended to use so that results are biased towards this location. Defaults to some central location in the region (see the `region` parameter).
        
    """
    url = f"https://local-business-data.p.rapidapi.com/search"
    querystring = {'query': query, }
    if x_user_agent:
        querystring['X-User-Agent'] = x_user_agent
    if region:
        querystring['region'] = region
    if business_status:
        querystring['business_status'] = business_status
    if subtypes:
        querystring['subtypes'] = subtypes
    if verified:
        querystring['verified'] = verified
    if lat:
        querystring['lat'] = lat
    if limit:
        querystring['limit'] = limit
    if language:
        querystring['language'] = language
    if zoom:
        querystring['zoom'] = zoom
    if fields:
        querystring['fields'] = fields
    if lng:
        querystring['lng'] = lng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_photos(business_id: str, limit: int=5, fields: str=None, region: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get business photos by Business Id."
    business_id: Unique Business Id. Accepts google_id / business_id and google_place_id. Examples of valid ids: `0x880fd393d427a591:0x8cba02d713a995ed`, `ChIJkaUn1JPTD4gR7ZWpE9cCuow`.
        limit: Maximum number of business photos to return (1-10000).

**Default**: *`20`*
        fields: A comma separated list of review fields to include in the response (field projection). By default all fields are returned.

**e.g.** `type,photo_url`
        region: Query Google Maps from a particular region or country. For a list of supported region/country codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (Alpha-2 code).

**Default:** *`us`*
        
    """
    url = f"https://local-business-data.p.rapidapi.com/business-photos"
    querystring = {'business_id': business_id, }
    if limit:
        querystring['limit'] = limit
    if fields:
        querystring['fields'] = fields
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_reviews(business_id: str, language: str='en', fields: str=None, sort_by: str=None, region: str='us', limit: int=5, offset: int=None, query: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all / paginate business reviews by Business Id."
    business_id: Unique Business Id. Accepts google_id / business_id and google_place_id. Examples of valid ids: `0x880fd393d427a591:0x8cba02d713a995ed`, `ChIJkaUn1JPTD4gR7ZWpE9cCuow`.
        language: Set the language of the results. For a list of supported language codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes .

**Default**: *`en`*
        fields: A comma separated list of review fields to include in the response (field projection). By default all fields are returned.

**e.g.** `review_id,review_text,rating`
        sort_by: How to sort the reviews in the results.

**Allowed values**: *`most_relevant, newest, highest_ranking, lowest_ranking`*.

**Default**: *`most_relevant`*
        region: Query Google Maps from a particular region or country. For a list of supported region/country codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (Alpha-2 code).

**Default:** *`us`*
        limit: Maximum number of business reviews to return (1-150).

**Default**: *`20`*
        offset: Number of business reviews to skip (for pagination/scrolling).

**Default**: *`0`*
        query: Return reviews matching a text query.
        
    """
    url = f"https://local-business-data.p.rapidapi.com/business-reviews"
    querystring = {'business_id': business_id, }
    if language:
        querystring['language'] = language
    if fields:
        querystring['fields'] = fields
    if sort_by:
        querystring['sort_by'] = sort_by
    if region:
        querystring['region'] = region
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_in_area(lat: str, zoom: str, query: str, lng: str, x_user_agent: str=None, region: str='us', language: str='en', subtypes: str=None, fields: str=None, limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search businesses in a specific geographic area defined by a center coordinate point and zoom level. To see it in action, make a query on Google Maps, wait for the results to show, move the map or change the zoom and click "Search this area" at the top."
    lat: Latitude of the center coordinate point of the area to search in.
        zoom: Zoom level on which to make the search (the search area / viewport is determined by lat, lng and zoom on a 1000x1000 screen).
        query: Search query / keyword.

**e.g.** `Bars and pubs`
**e.g.** `Plumbers`
        lng: Longitude of the center coordinate point of the area to search in.
        x_user_agent: Device type for the search. Default `desktop`.
        region: Query Google Maps from a particular region or country. For a list of supported region/country codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (Alpha-2 code).

**Default:** *`us`*
        language: Set the language of the results. For a list of supported language codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes .

**Default**: *`en`*
        subtypes: Find businesses with specific subtypes, specified as a comma separated list of types (business categories). For the complete list of types, see https://daltonluka.com/blog/google-my-business-categories.

**e.g.** `Plumber,Carpenter,Electrician`
**e.g.** `Night club,Dance club,Bar,Pub`
        fields: A comma separated list of business fields to include in the response (field projection). By default all fields are returned.

**e.g.** `business_id,type,phone_number,full_address`
        limit: Maximum number of businesses to return (1-500).

**Default**: *`20`*
        
    """
    url = f"https://local-business-data.p.rapidapi.com/search-in-area"
    querystring = {'lat': lat, 'zoom': zoom, 'query': query, 'lng': lng, }
    if x_user_agent:
        querystring['X-User-Agent'] = x_user_agent
    if region:
        querystring['region'] = region
    if language:
        querystring['language'] = language
    if subtypes:
        querystring['subtypes'] = subtypes
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_nearby(query: str, lng: int, lat: int, x_user_agent: str=None, language: str='en', subtypes: str=None, fields: str=None, limit: str='20', region: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search businesses near by specific geographic coordinates. To see it in action, right click on a specific point in the map on Google Maps and select "Search nearby", enter a query and search."
    query: Search query / keyword.

**e.g.** `Bars and pubs`
**e.g.** `Plumbers`
        lng: Longitude of the geographic coordinates to search near by.
        lat: Latitude of the geographic coordinates to search near by.
        x_user_agent: Device type for the search. Default `desktop`.
        language: Set the language of the results. For a list of supported language codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes .

**Default**: *`en`*
        subtypes: Find businesses with specific subtypes, specified as a comma separated list of types (business categories). For the complete list of types, see https://daltonluka.com/blog/google-my-business-categories.

**e.g.** `Plumber,Carpenter,Electrician`
**e.g.** `Night club,Dance club,Bar,Pub`
        fields: A comma separated list of business fields to include in the response (field projection). By default all fields are returned.

**e.g.** `business_id,type,phone_number,full_address`
        limit: Maximum number of businesses to return (1-500).

**Default**: *`20`*
        region: Query Google Maps from a particular region or country. For a list of supported region/country codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (Alpha-2 code).

**Default:** *`us`*
        
    """
    url = f"https://local-business-data.p.rapidapi.com/search-nearby"
    querystring = {'query': query, 'lng': lng, 'lat': lat, }
    if x_user_agent:
        querystring['X-User-Agent'] = x_user_agent
    if language:
        querystring['language'] = language
    if subtypes:
        querystring['subtypes'] = subtypes
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(query: str, language: str='en', region: str='us', coordinates: str='37.381315,-122.046148', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns place/address, business and query predictions for text-based geographic queries."
    query: Free-text geographic query.
        language: Set the language of the results. For a list of supported language codes see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 .
Default: `en`
        region: Return results biased to a particular region. For a list of supported region/country codes see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (Alpha-2 code).
Default: `us`
        coordinates: Geographic coordinates of the location from which the query is applied - recommended to use so that results are biased towards this location. Defaults to some central location in the region (see the `region` parameter).
        
    """
    url = f"https://local-business-data.p.rapidapi.com/autocomplete"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    if region:
        querystring['region'] = region
    if coordinates:
        querystring['coordinates'] = coordinates
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

