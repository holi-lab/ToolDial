import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def google_search(q: str, flatten_results: str=None, include_answer_box: str=None, hl: str=None, knowledge_graph_id: str=None, num: int=None, tbs: str=None, time_period_max: str=None, safe: str=None, lr: str=None, nfpr: str=None, page: int=None, filter: str=None, google_domain: str='google.com', time_period_min: str=None, cr: str=None, time_period: str=None, gl: str=None, location: str='united states', location_auto: bool=None, uule: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Search Parameters are applicable when making a request to the Search API to retrieve Google search results for a given search term. The search term is specified in the `q` parameter. The `location` your search is run from is determined by the location parameter, which can be populated with a `full_name` value from the Locations API."
    q: The keyword you want to use to perform the search.

        flatten_results: Can be set to `true` or `false`. Determines whether Scale SERP flattens the `inline_videos`, `inline_images`, `inline_tweets`, `top_stories` and `local_results` and shows them inline with the `organic_results`. This is useful if you want a simplified list of all of the results shown for an organic web search, irrespective of the type of result. When `flatten_results=true` then a new property type is added to each item in the `organic_results` array indicating the type of result (i.e. \\\\\\\\\\\\\\\"ad\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"`inline_tweets`\\\\\\\\\\\\\\\" etc).

        include_answer_box: Determines whether to include the answer box (sometimes called the \\\\\\\\\\\\\\\"featured snippet\\\\\\\\\\\\\\\") in the `organic_results` array and treat it as the first result. This may be desirable if you treat the result Bing displayed in the `answer_box` as the first organic result.

        knowledge_graph_id: The `knowledge_graph_id` request parameter sets the `kgmid` Google parameter. You can use this to prompt a specific knowledge graph to show in the results, an example would be `knowledge_graph_id=/m/0jg24`

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        filter: Determines if the filters for `Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_minor` `time_period_max` parameters to define the `custom` time period.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'q': q, }
    if flatten_results:
        querystring['flatten_results'] = flatten_results
    if include_answer_box:
        querystring['include_answer_box'] = include_answer_box
    if hl:
        querystring['hl'] = hl
    if knowledge_graph_id:
        querystring['knowledge_graph_id'] = knowledge_graph_id
    if num:
        querystring['num'] = num
    if tbs:
        querystring['tbs'] = tbs
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if safe:
        querystring['safe'] = safe
    if lr:
        querystring['lr'] = lr
    if nfpr:
        querystring['nfpr'] = nfpr
    if page:
        querystring['page'] = page
    if filter:
        querystring['filter'] = filter
    if google_domain:
        querystring['google_domain'] = google_domain
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if cr:
        querystring['cr'] = cr
    if time_period:
        querystring['time_period'] = time_period
    if gl:
        querystring['gl'] = gl
    if location:
        querystring['location'] = location
    if location_auto:
        querystring['location_auto'] = location_auto
    if uule:
        querystring['uule'] = uule
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places_and_maps_search(q: str, search_type: str, location: str=None, hl: str=None, google_domain: str=None, num: int=None, page: int=None, nfpr: str=None, uule: str=None, gl: str=None, location_auto: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scale SERP parses Google Local & Maps results (i.e. local business listings) when the `search_type` parameter is set to `search_type=places`. When this parameter is set Scale SERP provides a `places_results` array in the result JSON that contains the Places or Maps results.
		
		**Google Local vs. Google Maps**
		`search_type=places` requests can be executed in two modes to retrieve data from a Google Local results page or from a Google Maps results page.
		
		You set the Location of the search_type=places request using the location request parameter and this can be expressed as either a text location name from the Locations API (which will result in a Google Local result), or as a latitude, longitude and zoom level (which will result in a Google Maps result).
		
		If the `location` parameter is set to a latitude, longitude and zoom value - for example `location=lat:43.437677,lon:-3.8392765,zoom:15` then results will be returned from a Google Maps page."
    location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

**Google Maps**
If the location parameter is set to a latitude, longitude and zoom value - for example `location=lat:43.437677,lon:-3.8392765,zoom:15` then results will be returned from a Google Maps page.

**Formatting the location parameter for Google Maps requests**
To get Google Maps results based on latitude and longitude coordinates you should specify your location parameter in the form `location=lat:43.437677,lon:-3.8392765,zoom:15 `where `43.437677` is your `latitude` value, `-3.8392765` is your `longitude` value and `15` is your `zoom` value.

Delimit each section with a comma `,`, and delimit each name/value pair with a colon `:`.

Valid zoom values are between `3` (maximum zoom-out) and `21` (maximum zoom-in).

Google can return results outside the bounds of the zoom level in some instances. If this behaviour is not desirable then specifying `strict:true` in the location parameter will only return results within the current zoom level. Eg. `lat:39.58467741051493,lon:-0.6752313712718961,zoom:15,strict:true`
        num: **Note** the maximum number of results per page Google allow, for Places results, is `20`.

Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

**Note** that the number of results per page is fixed to **20** and that the `num` parameter has no effect when running Google Maps requests.
        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'q': q, 'search_type': search_type, }
    if location:
        querystring['location'] = location
    if hl:
        querystring['hl'] = hl
    if google_domain:
        querystring['google_domain'] = google_domain
    if num:
        querystring['num'] = num
    if page:
        querystring['page'] = page
    if nfpr:
        querystring['nfpr'] = nfpr
    if uule:
        querystring['uule'] = uule
    if gl:
        querystring['gl'] = gl
    if location_auto:
        querystring['location_auto'] = location_auto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places_and_maps_reviews(search_type: str, hl: str='place_reviews', data_id: str=None, topic_id: str=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Reviews Parameters are applicable when making a request with `search_type=place_reviews` to retrieve place reviews for a Place. The Place is specified in the `data_id` parameter, `data_id` values are returned from `search_type=places` Places requests.
		
		**Place Review Topic IDs**
		If shown, a `search_type=place_reviews` will return a topics array containing review topics. You can pass a place reviews topic ID in to the optional `topic_id` parameter to refine your place reviews request to just reviews related to that topic.
		
		**Place Reviews Pagination**
		Place Review results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `search_type=place_reviews` result will return a `next_page_token` in its' pagination object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of place reviews results."
    data_id: The `data_id` of the Place to retrieve place reviews for `data_id` values are returned in Places requests.

        topic_id: An optional topic ID to limit the returned reviews to just those related to the given topic. `topic_id` values, if shown for the given `data_id`, are returned in the topics array in all `search_type=place_reviews` results.

        sort_by: Determines how place reviews are sorted, valid values are `relevance`, `date`, `highest_rating` and `lowest_rating`. Defaults to `relevance`.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, }
    if hl:
        querystring['hl'] = hl
    if data_id:
        querystring['data_id'] = data_id
    if topic_id:
        querystring['topic_id'] = topic_id
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places_and_maps_photos(data_cid: str, search_type: str, data_id: str, next_page_token: str=None, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Photos Parameters are applicable when making a request with `search_type=place_photos` to retrieve place photos for a Place. The Place is specified in the `data_id` parameter, `data_id` values are returned from `search_type=places` Places requests.
		
		`Place Photo Category IDs`
		If shown, a `search_type=place_photos` will return a categories array containing photo categories. You can pass a place photos category ID in to the optional `category_id` parameter to refine your place photos request to just photos belonging to that category.
		
		**Place Photos Pagination**
		Place Photo results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `search_type=place_photos` result will return a `next_page_token` in its' pagination object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of place photo results."
    data_cid: The `data_cid` (a standard Google Place identifier) of the Place to retrieve place details for. `data_cid` values are returned in Places requests.

**Note** that either a `data_id` or `data_cid` identifier must be supplied.
        data_id: The `data_id` of the Place to retrieve place details for 'data_id'[ values are returned in Places requests.

**Note** that either a `data_id` or `data_cid` identifier must be supplied.
        next_page_token: Place Photo results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `search_type=place_photos` result will return a `next_page_token` in its' pagination object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of place photo results.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'data_cid': data_cid, 'search_type': search_type, 'data_id': data_id, }
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places_and_maps_details(data_id: str, data_cid: str, search_type: str, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Details Parameters are applicable when making a request with `search_type=place_details` to retrieve place details for a Place. The Place is specified in the `data_id` parameter, data_id values are returned from `search_type=places` Places requests.
		
		If the `data_id` is not available you may also request a place_details request using a `data_cid` (a standardised Google Place identifier you may already have access to). However, `data_id` is the recommended identifier to use."
    data_id: The `data_id` of the Place to retrieve place details for 'data_id'[ values are returned in Places requests.

**Note** that either a `data_id` or `data_cid` identifier must be supplied.
        data_cid: The `data_cid` (a standard Google Place identifier) of the Place to retrieve place details for. `data_cid` values are returned in Places requests.

**Note** that either a `data_id` or `data_cid` identifier must be supplied.
        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'data_id': data_id, 'data_cid': data_cid, 'search_type': search_type, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places_and_maps_products(search_type: str, data_cid: str, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Products Parameters are applicable when making a request with `search_type=place_products` to retrieve place products for a Place. The Place is specified in the `data_cid` parameter, `data_cid` values are returned from `search_type=places` Places requests.
		
		**`data_cid` IDs used in `place_products` requests**
		Note that `place_products` requests differ from `place_reviews` ([docs](https://www.scaleserp.com/docs/search-api/searches/google/place-reviews)) and `place_photos` ([docs](https://www.scaleserp.com/docs/search-api/searches/google/place-photos)) requests in that it required a `data_cid` instead of a data_id request parameter.
		
		Both `data_cid` and `data_id` IDs are returned in Places requests requests."
    data_cid: The `data_cid` of the Place to retrieve place details for 'data_cid' values are returned in Places requests.
        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'data_cid': data_cid, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_news_search(search_type: str, q: str, filter: str=None, exclude_if_modified: bool=None, page: int=2, cr: str=None, hl: str=None, location: str=None, google_domain: str='google.com', time_period_min: str=None, time_period: str=None, location_auto: bool=None, num: int=None, lr: str=None, tbs: str=None, sort_by: str=None, nfpr: str=None, gl: str=None, safe: str=None, uule: str=None, time_period_max: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google News Parameters are applicable when making a request with `search_type=news` to retrieve news results for a given search term. The search term is specified in the q parameter and the optional `location` parameter can be used to geo-locate the news request (locations can be retrieved via the Locations API).
		
		You can use the `time_period` request parameter to refine your news search to specific time periods.
		
		If you wish to exclude news results where Google have modified the original query set `exclude_if_modified=true` in your request parameters."
    q: The keyword you want to use to perform the News search.

        filter: Determines if the filters for `Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        exclude_if_modified: Determines whether the news_results returned when Google modifies the original query. Set to `true` to exclude `news_results` when Google modifies the original query or `false` (the default) to include `news_results` when Google modifies the original query.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_minor` `time_period_max` parameters to define the `custom` time period.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        sort_by: Determines how results are sorted, valid values are `relevance` or `date`. Note that when `sort_by=date` the show_duplicates parameter can be used.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'q': q, }
    if filter:
        querystring['filter'] = filter
    if exclude_if_modified:
        querystring['exclude_if_modified'] = exclude_if_modified
    if page:
        querystring['page'] = page
    if cr:
        querystring['cr'] = cr
    if hl:
        querystring['hl'] = hl
    if location:
        querystring['location'] = location
    if google_domain:
        querystring['google_domain'] = google_domain
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if time_period:
        querystring['time_period'] = time_period
    if location_auto:
        querystring['location_auto'] = location_auto
    if num:
        querystring['num'] = num
    if lr:
        querystring['lr'] = lr
    if tbs:
        querystring['tbs'] = tbs
    if sort_by:
        querystring['sort_by'] = sort_by
    if nfpr:
        querystring['nfpr'] = nfpr
    if gl:
        querystring['gl'] = gl
    if safe:
        querystring['safe'] = safe
    if uule:
        querystring['uule'] = uule
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations(q: str, page: str=None, country_code: str=None, is_id: str=None, type: str=None, limit: str=None, parent_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Scale SERP Locations API allows you to search for Scale SERP supported Google search locations. You can supply the `full_name` returned by the Locations API as the `location` parameter in a Search API query to retrieve search results localized to that location."
    q: The query to use to search for locations, i.e. new york or mumbai.

i.e. `London`
        page: Specifies the page of results to retrieve from `0` (first page) to a maximum page number of `100` (Scale SERP does not support paging beyond the 100th page). Defaults to `0`. Use in conjunction with the limit parameter to limit the number of results returned per page.

        country_code: Limits the results to locations in specific countries. See [supported countries](https://www.scaleserp.com/docs/search-api/reference/google-countries) for a full list of all supported values. Multiple countries should be presented as a comma separated list (i.e. `country_code=de,fr,us` for Germany, France and the United States).

        is_id: The `id` of a specific location (location ids are returned with all results from the Locations API).

        type: Limits the results to a specific type of location. Valid values are: `city`, `country`, `county`, `dma_region`, `municipality`, `neighborhood`, `state`, `postal_code` or `province`. Multiple location types should be presented as a comma separated list (i.e.`type=state,city` to include Location results that are of type `state` or `city`).

        limit: Limits the number of search results returned per page. Defaults to `10`. The maximum number of location results returned per page is `100`. Use in conjunction with the the `page` parameter to implement pagination.

        parent_id: Limits the search to child locations of the given `parent_id`. Useful for retrieving all sub-locations of a given location (all cities within a country, for example).

        
    """
    url = f"https://scale-serp.p.rapidapi.com/locations"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    if country_code:
        querystring['country_code'] = country_code
    if is_id:
        querystring['id'] = is_id
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    if parent_id:
        querystring['parent_id'] = parent_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_reviews(product_type: str, search_type: str, product_id: str, next_page_token: str=None, gl: str=None, sort_by: str=None, google_domain: str=None, location: str=None, location_auto: bool=None, hl: str=None, uule: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Reviews Parameters are applicable when making a request with `search_type=product` and `product_type=reviews` to retrieve product reviews results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a location parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a location that matches the location that was used to retrieve the `product_id` in the original Google Shopping search request.
		
		**Place Reviews Pagination**
		Product Reviews results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `product_type`=reviews result will return a `next_page_token` in its' `product_results.pagination` object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of Product Reviews results."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by Google Shopping search requests.

        next_page_token: Product Reviews results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `product_type=reviews` result will return a `next_page_token` in its' `product_results.pagin`ation object. This next_page_token can be passed in to the `next_page_token` request parameter to retrieve the next page of Product Reviews results.

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.scaleserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        sort_by: 
Sets the sort ordering of the product reviews returned. Valid values are:
`relevance`
Sort product reviews results by relevance, the default.
`date`
Sort product reviews results by date, most recent first.
        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'product_type': product_type, 'search_type': search_type, 'product_id': product_id, }
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    if gl:
        querystring['gl'] = gl
    if sort_by:
        querystring['sort_by'] = sort_by
    if google_domain:
        querystring['google_domain'] = google_domain
    if location:
        querystring['location'] = location
    if location_auto:
        querystring['location_auto'] = location_auto
    if hl:
        querystring['hl'] = hl
    if uule:
        querystring['uule'] = uule
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_specificiations(product_id: str, search_type: str, product_type: str, location: str=None, gl: str=None, page: str=None, location_auto: bool=None, hl: str=None, google_domain: str=None, uule: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Specifications Parameters are applicable when making a request with `search_type=product` and product_type=specifications to retrieve product specifications results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a `location` parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		`Products & Location`
		Google Product pages are highly location-sensitive so it is important that you specify a `location` that matches the `location` that was used to retrieve the `product_id` in the original Google Shopping search request."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by Google Shopping search requests.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.scaleserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        page: Determines the page of results to return, defaults to `1`.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'product_id': product_id, 'search_type': search_type, 'product_type': product_type, }
    if location:
        querystring['location'] = location
    if gl:
        querystring['gl'] = gl
    if page:
        querystring['page'] = page
    if location_auto:
        querystring['location_auto'] = location_auto
    if hl:
        querystring['hl'] = hl
    if google_domain:
        querystring['google_domain'] = google_domain
    if uule:
        querystring['uule'] = uule
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_online_sellers(search_type: str, product_id: str, product_type: str, product_condition_used: str=None, product_condition_new: str=None, product_free_shipping: str=None, page: str=None, hl: str=None, google_domain: str=None, gl: str=None, location: str=None, sort_by: str=None, uule: str=None, location_auto: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Online Sellers Parameters are applicable when making a request with `search_type=product` and `product_type=sellers_online` to retrieve product online sellers results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a `location` parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a `location` that matches the `location` that was used to retrieve the `product_id` in the original Google Shopping search request."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by Google Shopping search requests.

        product_condition_used: Determines whether to filter to only used (non-new) products. Valid values are `true` or `false`.

        product_condition_new: Determines whether to filter to only new (non-used) products. Valid values are `true` or `false`.

        product_free_shipping: Determines whether to filter to only products with free shipping. Valid values are `true` or `false`.

        page: Determines the page of results to return, defaults to `1`.

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.scaleserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        sort_by: 
Sets the sort ordering of the product online sellers returned. Valid values are:
`base_price`
Sort product online sellers results by base price.
`total_price`
Sort product online sellers results by total price.
`promotion`
Sort product online sellers results by current promotion deals (special offers).
`seller_rating`
Sort product online sellers results by seller rating (high to low).
        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'product_id': product_id, 'product_type': product_type, }
    if product_condition_used:
        querystring['product_condition_used'] = product_condition_used
    if product_condition_new:
        querystring['product_condition_new'] = product_condition_new
    if product_free_shipping:
        querystring['product_free_shipping'] = product_free_shipping
    if page:
        querystring['page'] = page
    if hl:
        querystring['hl'] = hl
    if google_domain:
        querystring['google_domain'] = google_domain
    if gl:
        querystring['gl'] = gl
    if location:
        querystring['location'] = location
    if sort_by:
        querystring['sort_by'] = sort_by
    if uule:
        querystring['uule'] = uule
    if location_auto:
        querystring['location_auto'] = location_auto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product(search_type: str, product_id: str, product_free_shipping: bool=None, product_condition_new: bool=None, product_condition_used: bool=None, uule: str=None, gl: str=None, hl: str=None, location_auto: bool=None, google_domain: str=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Parameters are applicable when making a request with `search_type=product` to retrieve product results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a `location` parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by Google Shopping search requests.

        product_free_shipping: Determines whether to filter to only products with free shipping. Valid values are `true` or `false`.

        product_condition_new: Determines whether to filter to only new (non-used) products. Valid values are `true` or `false`.

        product_condition_used: Determines whether to filter to only used (non-new) products. Valid values are `true` or `false`.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'product_id': product_id, }
    if product_free_shipping:
        querystring['product_free_shipping'] = product_free_shipping
    if product_condition_new:
        querystring['product_condition_new'] = product_condition_new
    if product_condition_used:
        querystring['product_condition_used'] = product_condition_used
    if uule:
        querystring['uule'] = uule
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if location_auto:
        querystring['location_auto'] = location_auto
    if google_domain:
        querystring['google_domain'] = google_domain
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_shopping_search(search_type: str, q: str, shopping_merchants: str=None, shopping_price_max: str=None, shopping_condition: str=None, shopping_price_min: str=None, sort_by: str=None, shopping_buy_on_google: bool=None, shopping_filter: str=None, location: str=None, uule: str=None, google_domain: str=None, page: int=None, location_auto: bool=None, num: int=None, gl: str=None, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Shopping Parameters are applicable when making a request with `search_type=shopping` to retrieve Shopping results for a given search term. The search term is specified in the `q` parameter and the optional `location` parameter can be used to geo-locate the shopping request (locations can be retrieved via the Locations API)."
    q: The keyword you want to use to perform the search.

        shopping_merchants: A comma-separated list of merchant IDs to retrieve shopping results. Merchant ID's can be found in the` merchagg:` section of any Google Shopping URL.

        shopping_price_max: The maximum price of products returned. For example `shopping_price_max=29.99`

        shopping_condition: The condition of products returned. Can be set to `new` or `used`.

        shopping_price_min: The minimum price of products. For example `shopping_price_min=4.99`

        sort_by: Sets the sort ordering of the shopping results returned. Valid values are:
`relevance`
Sort shopping results by relevance to the search term supplied in the q parameter, the default.

`price_low_to_high`
Sort shopping results by lowest to highest price.

`price_high_to_low`
Sort shopping results by highest to lowest price.

`review_score`
Sort shopping results by review score, highest review score first.
        shopping_buy_on_google: Determines whether the \"Buy on Google\" option is selected when running a `search_type=shopping` search. Valid values are `true` or `false`.

        shopping_filter: A shopping filter (i.e. \"Brand\") to filter the results to. Shopping filter values are returned in the filters property of the shopping response.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'q': q, }
    if shopping_merchants:
        querystring['shopping_merchants'] = shopping_merchants
    if shopping_price_max:
        querystring['shopping_price_max'] = shopping_price_max
    if shopping_condition:
        querystring['shopping_condition'] = shopping_condition
    if shopping_price_min:
        querystring['shopping_price_min'] = shopping_price_min
    if sort_by:
        querystring['sort_by'] = sort_by
    if shopping_buy_on_google:
        querystring['shopping_buy_on_google'] = shopping_buy_on_google
    if shopping_filter:
        querystring['shopping_filter'] = shopping_filter
    if location:
        querystring['location'] = location
    if uule:
        querystring['uule'] = uule
    if google_domain:
        querystring['google_domain'] = google_domain
    if page:
        querystring['page'] = page
    if location_auto:
        querystring['location_auto'] = location_auto
    if num:
        querystring['num'] = num
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places_and_maps_posts(search_type: str, data_cid: str, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Posts Parameters are applicable when making a request with `search_type=place_posts` to retrieve place posts for a Place. The Place is specified in the `data_id` parameter, `data_id` values are returned from `search_type=places` Places requests."
    data_cid: The `data_cid` of the Place to retrieve place details for 'data_cid' values are returned in Places requests.
        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'data_cid': data_cid, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_autocomplete(autocomplete_search_index: str=None, google_domain: str=None, location: str=None, location_auto: bool=None, uule: str='videos', hl: str=None, gl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Autocomplete Parameters are applicable when making a request with `search_type=autocomplete` to retrieve autocomplete results for a given search term. The keyword searched can be specified in the `q` parameter and the optional `location` parameter can be used to geo-locate the autocomplete request (locations can be retrieved via the Locations API)."
    autocomplete_search_index: Optional parameter to determine on which index of the string provided in the `q` param the autocomplete operation is requested. Different autocomplete results are returned by Google depending on the index within the search term the user currently has their cursor set to.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {}
    if autocomplete_search_index:
        querystring['autocomplete_search_index'] = autocomplete_search_index
    if google_domain:
        querystring['google_domain'] = google_domain
    if location:
        querystring['location'] = location
    if location_auto:
        querystring['location_auto'] = location_auto
    if uule:
        querystring['uule'] = uule
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_scholar_search(search_type: str, q: str, scisbd: int=None, scholar_year_max: str=None, scholar_include_citations: bool=None, scholar_patents_courts: str=None, gl: str=None, scholar_year_min: str=None, location_auto: bool=None, page: int=None, lr: str=None, hl: str=None, google_domain: str=None, cr: str='scholar', uule: str='videos', location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Scholar Parameters are applicable when making a request with `search_type=scholar` to retrieve scholar results for a given search term. The search term is specified in the q parameter and the optional `location` parameter can be used to geo-locate the scholar request (locations can be retrieved via the Locations API)."
    q: The keyword you want to use to perform the Images search.

        scisbd: The Scholar scisbd parameter can be set to `1` or `2`.

        scholar_year_max: Determines the year from which results should be included when `search_type=scholar`. For example, if you set `scholar_year_max=2019` the results after 2019 will be omitted. This parameter can be combined with the `scholar_year_min` parameter.

        scholar_include_citations: Determines whether to include citations or not when the `search_type` parameter is set to scholar. Valid values are `true` or `false`. Defaults to `true`. Drives the following option:

        scholar_patents_courts: The scholar_patents_courts parameter can be used either as both a way of limiting the Scholar search or as a filter (this is due to how Google implements this parameter itself).

**As a Filter the valid values are:**
`0` (default) - include patents
`1` - exclude patents

**As a way of limiting a Scholar search the valid values are:**
`4` - case law (US courts only). This will select all the State and Federal courts.

To select specific courts, see the [full list of supported Google Scholar courts](https://www.scaleserp.com/docs/search-api/reference/google-scholar-courts).

For example, `scholar_patents_courts=4,33,192` `4` is the required value and should always be in the first position, `33` selects all New York courts and `192` selects Tax Court. Values must be separated by a comma.
        scholar_year_min: Determines the year from which results should be included when `search_type=scholar`. For example, if you set `scholar_year_min=2018` the results before 2018 will be omitted. This parameter can be combined with the `scholar_year_max` parameter.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'q': q, }
    if scisbd:
        querystring['scisbd'] = scisbd
    if scholar_year_max:
        querystring['scholar_year_max'] = scholar_year_max
    if scholar_include_citations:
        querystring['scholar_include_citations'] = scholar_include_citations
    if scholar_patents_courts:
        querystring['scholar_patents_courts'] = scholar_patents_courts
    if gl:
        querystring['gl'] = gl
    if scholar_year_min:
        querystring['scholar_year_min'] = scholar_year_min
    if location_auto:
        querystring['location_auto'] = location_auto
    if page:
        querystring['page'] = page
    if lr:
        querystring['lr'] = lr
    if hl:
        querystring['hl'] = hl
    if google_domain:
        querystring['google_domain'] = google_domain
    if cr:
        querystring['cr'] = cr
    if uule:
        querystring['uule'] = uule
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_images_search(q: str, search_type: str, images_type: str=None, hl: str=None, time_period_min: str=None, gl: str=None, time_period: str=None, location_auto: bool=None, images_page: str=None, safe: str=None, google_domain: str=None, images_usage: str=None, images_size: str=None, uule: str=None, tbs: str=None, time_period_max: str=None, cr: str=None, images_color: str=None, lr: str=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Image Parameters are applicable when making a request with `search_type=images` to retrieve image results for a given search term. The search term is specified in the q parameter and the optional `location` parameter can be used to geo-locate the image request (locations can be retrieved via the Locations API).
		You can use the `time_period` request parameter to refine your news search to specific time periods.
		
		If you wish to exclude news results where Google have modified the original query set `exclude_if_modified=true` in your request parameters."
    q: The keyword you want to use to perform the Images search.

        images_type: Allows you to set the size of the images returned when `search_type=images`. Allows you to set the type of the images returned when `search_type=images`. Valid values are `clipart`, `line_drawing`, or `gif`.

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_minor` `time_period_max` parameters to define the `custom` time period.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        images_page: Determines the page of results to get when `search_type=images`. For images Google will return 100 results per page (a Google-imposed limit). Use `images_page` to specify the page of results to retrieve, for example is `images_page=2` the API will return images 100-199.

        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        images_usage: Allows you to specify the usage rights of the images returned when `search_type=images`. Valid values are `reuse_with_modification`, `reuse`, `non_commercial_reuse_with_modification` or `non_commercial_reuse`.

        images_size: Allows you to set the size of the images returned when `search_type=images`. Valid values are `large`, `medium`, or `icon`.

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        images_color: Allows you to set the color of the images returned when `search_type=images`. Valid values are `any`, `black_and_white`, `transparent`, `red`, `orange`, `yellow`, `green`, `teal`, `blue`, `purple`, `pink`, `white`, `gray`, `black` or `brown`.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'q': q, 'search_type': search_type, }
    if images_type:
        querystring['images_type'] = images_type
    if hl:
        querystring['hl'] = hl
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if gl:
        querystring['gl'] = gl
    if time_period:
        querystring['time_period'] = time_period
    if location_auto:
        querystring['location_auto'] = location_auto
    if images_page:
        querystring['images_page'] = images_page
    if safe:
        querystring['safe'] = safe
    if google_domain:
        querystring['google_domain'] = google_domain
    if images_usage:
        querystring['images_usage'] = images_usage
    if images_size:
        querystring['images_size'] = images_size
    if uule:
        querystring['uule'] = uule
    if tbs:
        querystring['tbs'] = tbs
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if cr:
        querystring['cr'] = cr
    if images_color:
        querystring['images_color'] = images_color
    if lr:
        querystring['lr'] = lr
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_video_search(q: str, search_type: str, nfpr: int=None, safe: str=None, tbs: str=None, time_period: str=None, hl: str=None, filter: int=None, page: int=None, location: str=None, time_period_max: str=None, cr: str=None, time_period_min: str=None, uule: str='videos', google_domain: str=None, lr: str=None, location_auto: bool=None, gl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Video Parameters are applicable when making a request with `search_type=videos` to retrieve video results for a given search term. The search term is specified in the `q` parameter and the optional location parameter can be used to geo-locate the videos request (locations can be retrieved via the Locations API)."
    q: The keyword you want to use to perform the Images search.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_minor` `time_period_max` parameters to define the `custom` time period.

        filter: Determines if the filters for `Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        location: Determines the geographic location in which the query is executed. You can enter any `location` as free-text, but if you choose one of the Scale SERP [built-in locations](https://www.scaleserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        location_auto: If the `location` field is set to a Scale SERP built-in [location](https://www.scaleserp.com/docs/locations-api/overview) from the [Locations API](https://www.scaleserp.com/docs/locations-api), and `location_auto` is set to true (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://scale-serp.p.rapidapi.com/search"
    querystring = {'q': q, 'search_type': search_type, }
    if nfpr:
        querystring['nfpr'] = nfpr
    if safe:
        querystring['safe'] = safe
    if tbs:
        querystring['tbs'] = tbs
    if time_period:
        querystring['time_period'] = time_period
    if hl:
        querystring['hl'] = hl
    if filter:
        querystring['filter'] = filter
    if page:
        querystring['page'] = page
    if location:
        querystring['location'] = location
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if cr:
        querystring['cr'] = cr
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if uule:
        querystring['uule'] = uule
    if google_domain:
        querystring['google_domain'] = google_domain
    if lr:
        querystring['lr'] = lr
    if location_auto:
        querystring['location_auto'] = location_auto
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scale-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

