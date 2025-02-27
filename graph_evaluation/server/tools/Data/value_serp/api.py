import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def google_search(q: str, include_answer_box: bool=None, page: int=None, flatten_results: bool=None, nfpr: int=None, tbs: str=None, num: int=None, time_period_max: str=None, safe: str=None, knowledge_graph_id: str=None, uule: str=None, lr: str=None, cr: str=None, filter: int=None, time_period_min: str=None, google_domain: str=None, time_period: str=None, location: str=None, hl: str=None, location_auto: bool=None, gl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Search Parameters are applicable when making a request to the Search API to retrieve Google search results for a given search term. The search term is specified in the q parameter. The location your search is run from is determined by the location parameter, which can be populated with a full_name value from the Locations API."
    q: The keyword you want to use to perform the search.
        include_answer_box: Determines whether to include the answer box (sometimes called the \\\\\\\\\\\\\\\"featured snippet\\\\\\\\\\\\\\\") in the `organic_results` array and treat it as the first result. This may be desirable if you treat the result Bing displayed in the `answer_box` as the first organic result.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the num parameter to implement pagination.
        flatten_results: Can be set to `true` or `false`. Determines whether VALUE SERP flattens the `inline_videos`, `inline_images`, `inline_tweets`, `top_stories` and `local_results` and shows them inline with the `organic_results`. This is useful if you want a simplified list of all of the results shown for an organic web search, irrespective of the type of result. When `flatten_results=true `then a new property type is added to each item in the `organic_results` array indicating the type of result (i.e. \\\\\\\\\\\\\\\"ad\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"inline_tweets\\\\\\\\\\\\\\\" etc).

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0 `(default) to include them.

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination. 

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        knowledge_graph_id: The `knowledge_graph_id` request parameter sets the `kgmid` Google parameter. You can use this to prompt a specific knowledge graph to show in the results, an example would be `knowledge_graph_id=/m/0jg24`

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. VALUE SERP automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.valueserp.com/docs/search-api/reference/google-lr-languages).

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.valueserp.com/docs/search-api/reference/google-cr-countries).

        filter: Determines if the filters for` Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com`.
        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_min` or `time_period_max` parameters to define the custom time period.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP[ built-in locations](https://www.valueserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).
        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location_auto: If the `location` field is set to a VALUE SERP [built-in location](https://www.valueserp.com/docs/locations-api) from the [Locations API](https://www.valueserp.com/docs/locations-api), and `location_auto` is set to `true` (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.
        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'q': q, }
    if include_answer_box:
        querystring['include_answer_box'] = include_answer_box
    if page:
        querystring['page'] = page
    if flatten_results:
        querystring['flatten_results'] = flatten_results
    if nfpr:
        querystring['nfpr'] = nfpr
    if tbs:
        querystring['tbs'] = tbs
    if num:
        querystring['num'] = num
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if safe:
        querystring['safe'] = safe
    if knowledge_graph_id:
        querystring['knowledge_graph_id'] = knowledge_graph_id
    if uule:
        querystring['uule'] = uule
    if lr:
        querystring['lr'] = lr
    if cr:
        querystring['cr'] = cr
    if filter:
        querystring['filter'] = filter
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if google_domain:
        querystring['google_domain'] = google_domain
    if time_period:
        querystring['time_period'] = time_period
    if location:
        querystring['location'] = location
    if hl:
        querystring['hl'] = hl
    if location_auto:
        querystring['location_auto'] = location_auto
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations(q: str, type: str=None, limit: str=None, parent_id: str=None, is_id: str=None, page: str=None, country_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The VALUE SERP Locations API allows you to search for VALUE SERP supported Google search locations. You can supply the `full_name` returned by the Locations API as the `location` parameter in a Search API query to retrieve search results localized to that location."
    q: The query to use to search for locations, i.e. new york or mumbai.

i.e. `London`
        type: Limits the results to a specific type of location. Valid values are: `city`, `country`, `county`, `dma_region`, `municipality`, `neighborhood`, `state`, `postal_code` or `province`. Multiple location types should be presented as a comma separated list (i.e.`type=state,city` to include Location results that are of type `state` or `city`).

        limit: Limits the number of search results returned per page. Defaults to `10`. The maximum number of location results returned per page is `100`. Use in conjunction with the the `page` parameter to implement pagination.

        parent_id: Limits the search to child locations of the given `parent_id`. Useful for retrieving all sub-locations of a given location (all cities within a country, for example).

        is_id: The `id` of a specific location (location ids are returned with all results from the Locations API).

        page: Specifies the page of results to retrieve from `0` (first page) to a maximum page number of `100` (Value SERP does not support paging beyond the 100th page). Defaults to `0`. Use in conjunction with the limit parameter to limit the number of results returned per page.

        country_code: Limits the results to locations in specific countries. See [supported countries](https://www.valueserp.com/docs/search-api/reference/google-countries) for a full list of all supported values. Multiple countries should be presented as a comma separated list (i.e. `country_code=de,fr,us` for Germany, France and the United States).

        
    """
    url = f"https://value-serp.p.rapidapi.com/locations"
    querystring = {'q': q, }
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    if parent_id:
        querystring['parent_id'] = parent_id
    if is_id:
        querystring['id'] = is_id
    if page:
        querystring['page'] = page
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_online_sellers(product_id: str, product_type: str, gl: str=None, location_auto: str=None, hl: str=None, uule: str=None, location: str=None, product_free_shipping: bool=None, google_domain: str=None, product_condition_new: bool=None, sort_by: str=None, product_condition_used: bool=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Online Sellers Parameters are applicable when making a request with `search_type=product` and `product_type=sellers_online` to retrieve product online sellers results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a location parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a location that matches the location that was used to retrieve the `product_id` in the original Google Shopping search request."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by [Google Shopping search ](https://www.valueserp.com/docs/search-api/results/google/shopping)requests.
        product_type: Should be set to `product_type=reviews`

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        location_auto: If the `location` field is set to a VALUE SERP built-in location from the Locations API, and `location_auto` is set to `true` (default) then the google_domain, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        uule: The Google UULE parameter - use to pass through a custom uule parameter to Google. VALUE SERP automatically generates the uule when you use the `location` parameter but we allow you to overwrite it directly by specifying a uule directly.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP `built-in locations `then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_autoparameter`).

Note that Google Product requests are highly location-sensitive. It's important that you set a `location` parameter for the geographic location in which the `product_id` was found.
        product_free_shipping: Determines whether to filter to only products with free shipping. Valid values are `true` or `false`.

        google_domain: The Google domain to use to run the query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com.`

        product_condition_new: Determines whether to filter to only new (non-used) products. Valid values are `true` or `false`.

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
        product_condition_used: Determines whether to filter to only used (non-new) products. Valid values are `true` or `false`.

        page: Determines the page of results to return, defaults to `1`.

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'product_id': product_id, 'product_type': product_type, }
    if gl:
        querystring['gl'] = gl
    if location_auto:
        querystring['location_auto'] = location_auto
    if hl:
        querystring['hl'] = hl
    if uule:
        querystring['uule'] = uule
    if location:
        querystring['location'] = location
    if product_free_shipping:
        querystring['product_free_shipping'] = product_free_shipping
    if google_domain:
        querystring['google_domain'] = google_domain
    if product_condition_new:
        querystring['product_condition_new'] = product_condition_new
    if sort_by:
        querystring['sort_by'] = sort_by
    if product_condition_used:
        querystring['product_condition_used'] = product_condition_used
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product(product_id: str, type: str, gl: str=None, hl: str=None, uule: str=None, location_auto: str=None, google_domain: str=None, location: str=None, product_condition_new: bool=None, product_condition_used: bool=None, product_free_shipping: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Parameters are applicable when making a request with `search_type=product` to retrieve product results for a given product ID. The product ID is specified in the `product_id `parameter and you should also specify a `location` parameter to geo-locate the request (locations can be retrieved via the [Locations API](https://www.valueserp.com/docs/locations-api/overview)).
		
		Google Product IDs are returned by Google Shopping search requests."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by [Google Shopping search ](https://www.valueserp.com/docs/search-api/results/google/shopping)requests.
        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        uule: The Google UULE parameter - use to pass through a custom uule parameter to Google. VALUE SERP automatically generates the uule when you use the `location` parameter but we allow you to overwrite it directly by specifying a uule directly.

        location_auto: If the `location` field is set to a VALUE SERP built-in location from the Locations API, and `location_auto` is set to `true` (default) then the google_domain, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        google_domain: The Google domain to use to run the query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com.`

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP `built-in locations `then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_autoparameter`).

Note that Google Product requests are highly location-sensitive. It's important that you set a `location` parameter for the geographic location in which the `product_id` was found.
        product_condition_new: Determines whether to filter to only new (non-used) products. Valid values are `true` or `false.`

        product_condition_used: Determines whether to filter to only used (non-new) products. Valid values are `true` or `false.`

        product_free_shipping: Determines whether to filter to only products with free shipping. Valid values are `true` or `false.`

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'product_id': product_id, 'type': type, }
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if uule:
        querystring['uule'] = uule
    if location_auto:
        querystring['location_auto'] = location_auto
    if google_domain:
        querystring['google_domain'] = google_domain
    if location:
        querystring['location'] = location
    if product_condition_new:
        querystring['product_condition_new'] = product_condition_new
    if product_condition_used:
        querystring['product_condition_used'] = product_condition_used
    if product_free_shipping:
        querystring['product_free_shipping'] = product_free_shipping
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_place_details(search_type: str, data_cid: str, data_id: str, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Details Parameters are applicable when making a request with search_type=place_details to retrieve place details for a Place. The Place is specified in the data_id parameter, data_id values are returned from search_type=places Places requests.
		
		If the data_id is not available you may also request a place_details request using a data_cid (a standardised Google Place identifier you may already have access to). However, data_id is the recommended identifier to use."
    data_cid: The `data_cid` (a standard Google Place identifier) of the Place to retrieve place details for. `data_cid` values are returned in [Places requests](https://www.valueserp.com/docs/search-api/searches/google/places).

Note that either a `data_id` or `data_cid` identifier must be supplied.
        data_id: The `data_id` of the Place to retrieve place details for. `data_id` values are returned in [Places requests](https://www.valueserp.com/docs/search-api/searches/google/places).

Note that either a `data_id` or `data_cid` identifier must be supplied.
        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'data_cid': data_cid, 'data_id': data_id, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_specifications(type: str, product_id: str, product_type: str, page: int=None, gl: str=None, hl: str=None, google_domain: str=None, uule: str=None, location: str=None, location_auto: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Specifications Parameters are applicable when making a request with `search_type=product` and `product_type=specifications` to retrieve product specifications results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a location parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a `location` that matches the location that was used to retrieve the `product_id` in the original Google Shopping search request."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by [Google Shopping search ](https://www.valueserp.com/docs/search-api/results/google/shopping)requests.
        product_type: Should be set to `product_type=reviews`

        page: Determines the page of results to return, defaults to `1`.

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        google_domain: The Google domain to use to run the query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com.`

        uule: The Google UULE parameter - use to pass through a custom uule parameter to Google. VALUE SERP automatically generates the uule when you use the `location` parameter but we allow you to overwrite it directly by specifying a uule directly.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP `built-in locations `then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_autoparameter`).

Note that Google Product requests are highly location-sensitive. It's important that you set a `location` parameter for the geographic location in which the `product_id` was found.
        location_auto: If the `location` field is set to a VALUE SERP built-in location from the Locations API, and `location_auto` is set to `true` (default) then the google_domain, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'type': type, 'product_id': product_id, 'product_type': product_type, }
    if page:
        querystring['page'] = page
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if google_domain:
        querystring['google_domain'] = google_domain
    if uule:
        querystring['uule'] = uule
    if location:
        querystring['location'] = location
    if location_auto:
        querystring['location_auto'] = location_auto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_reviews(product_id: str, product_type: str, next_page_token: str=None, hl: str=None, uule: str=None, gl: str=None, location: str=None, location_auto: str=None, sort_by: str=None, google_domain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Reviews Parameters are applicable when making a request with `search_type=product` and `product_type=reviews` to retrieve product reviews results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a location parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a location that matches the location that was used to retrieve the `product_id` in the original Google Shopping search request.
		
		**Place Reviews Pagination**
		Product Reviews results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `product_type=reviews` result will return a `next_page_token` in its' product_results.pagination object. This `next_page_token `can be passed in to the `next_page_token` request parameter to retrieve the next page of Product Reviews results."
    product_id: The Google Product ID to retrieve. Google Product IDs are returned by [Google Shopping search ](https://www.valueserp.com/docs/search-api/results/google/shopping)requests.
        product_type: Should be set to `product_type=reviews`

        next_page_token: Product Reviews results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `product_type=reviews` result will return a `next_page_token` in its `product_results.pagination` object. This `next_page_token` can be passed in to the next_page_token request parameter to retrieve the next page of Product Reviews results.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        uule: The Google UULE parameter - use to pass through a custom uule parameter to Google. VALUE SERP automatically generates the uule when you use the `location` parameter but we allow you to overwrite it directly by specifying a uule directly.

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP `built-in locations `then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_autoparameter`).

Note that Google Product requests are highly location-sensitive. It's important that you set a `location` parameter for the geographic location in which the `product_id` was found.
        location_auto: If the `location` field is set to a VALUE SERP built-in location from the Locations API, and `location_auto` is set to `true` (default) then the google_domain, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        sort_by: 
Sets the sort ordering of the product reviews returned. Valid values are:
`relevance`
Sort product reviews results by relevance, the default.
`date`
Sort product reviews results by date, most recent first.
        google_domain: The Google domain to use to run the query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com.`

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'product_id': product_id, 'product_type': product_type, }
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    if hl:
        querystring['hl'] = hl
    if uule:
        querystring['uule'] = uule
    if gl:
        querystring['gl'] = gl
    if location:
        querystring['location'] = location
    if location_auto:
        querystring['location_auto'] = location_auto
    if sort_by:
        querystring['sort_by'] = sort_by
    if google_domain:
        querystring['google_domain'] = google_domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_shopping(q: str, type: str, shopping_filters: str=None, shopping_condition: str=None, shopping_price_max: str=None, shopping_price_min: str=None, shopping_merchants: str=None, uule: str=None, google_domain: str=None, page: int=None, sort_by: str=None, gl: str=None, shopping_buy_on_google: bool=None, num: str=None, location_auto: bool=None, location: str=None, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Shopping Parameters are applicable when making a request with `search_type=shopping` to retrieve Shopping results for a given search term. The search term is specified in the `q` parameter and the optional location parameter can be used to geo-locate the shopping request (`locations` can be retrieved via the [Locations API](https://www.valueserp.com/docs/locations-api/overview))."
    q: The keyword you want to use to perform the Shopping search.
        shopping_filters: A shopping filter (i.e. \"Brand\") to filter the results to. Shopping filter values are returned in the `filters` property of the [shopping response](https://www.valueserp.com/docs/search-api/results/google/shopping).

        shopping_condition: The condition of products returned. Can be set to `new` or `used`.

        shopping_price_max: The maximum price of products. For example `shopping_price_max=4.99`

        shopping_price_min: The minimum price of products. For example `shopping_price_min=4.99`

        shopping_merchants: A comma-separated list of merchant IDs to retrieve shopping results. Merchant ID's can be found in the `merchagg:` section of any Google Shopping URL.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. VALUE SERP automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com`.
        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        sort_by: Sets the sort ordering of the shopping results returned. Valid values are:

`relevance`
Sort shopping results by relevance to the search term supplied in the q parameter, the default.
`price_low_to_high`
Sort shopping results by lowest to highest price.
`price_high_to_low`
Sort shopping results by highest to lowest price.
`review_score`
Sort shopping results by review score, highest review score first.
        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        shopping_buy_on_google: Determines whether the \\\"Buy on Google\\\" option is selected when running a `search_type=shopping` search. Valid values are `true` or false.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination.

Note the `num` parameter is not always honoured by Google for `search_type=shopping` requests. When using the num parameter you are requesting that number of results per page. There is no guarantee that that number of results will necessarily be served.
        location_auto: If the `location` field is set to a VALUE SERP [built-in location](https://www.valueserp.com/docs/locations-api) from the [Locations API](https://www.valueserp.com/docs/locations-api), and `location_auto` is set to `true` (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.
        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP[ built-in locations](https://www.valueserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).
        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'q': q, 'type': type, }
    if shopping_filters:
        querystring['shopping_filters'] = shopping_filters
    if shopping_condition:
        querystring['shopping_condition'] = shopping_condition
    if shopping_price_max:
        querystring['shopping_price_max'] = shopping_price_max
    if shopping_price_min:
        querystring['shopping_price_min'] = shopping_price_min
    if shopping_merchants:
        querystring['shopping_merchants'] = shopping_merchants
    if uule:
        querystring['uule'] = uule
    if google_domain:
        querystring['google_domain'] = google_domain
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if gl:
        querystring['gl'] = gl
    if shopping_buy_on_google:
        querystring['shopping_buy_on_google'] = shopping_buy_on_google
    if num:
        querystring['num'] = num
    if location_auto:
        querystring['location_auto'] = location_auto
    if location:
        querystring['location'] = location
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places(page: int=None, hl: str=None, location: str=None, gl: str=None, num: int=None, google_domain: str=None, location_auto: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "VALUE SERP parses Google Local & Maps results (i.e. local business listings) when the search_type parameter is set to search_type=places. When this parameter is set VALUE SERP provides a places_results array in the result JSON that contains the [Places or Maps results](https://www.valueserp.com/docs/search-api/results/google/places).
		
		**Google Local vs. Google Maps**
		`search_type=places` requests can be executed in two modes to retrieve data from a [Google Local results page](https://www.google.com/search?q=pizza&gl=us&hl=en&uule=w+CAIQICIgTWFuaGF0dGFuLE5ldyBZb3JrLFVuaXRlZCBTdGF0ZXM&tbm=lcl) or from a [Google Maps results page](https://www.google.com/maps/search/pizza/@43.437677,-3.8392765,15z).
		
		You set the Location of the `search_type=places` request using the `location` request parameter and this can be expressed as either a text location name from the [Locations API](https://www.valueserp.com/docs/locations-api/overview) (which will result in a [Google Local result](https://www.google.com/search?q=pizza&gl=us&hl=en&uule=w+CAIQICIgTWFuaGF0dGFuLE5ldyBZb3JrLFVuaXRlZCBTdGF0ZXM&tbm=lcl)), or as a latitude, longitude and zoom level (which will result in a [Google Maps result](https://www.google.com/maps/search/pizza/@43.437677,-3.8392765,15z))."
    page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location: **Google Local**

Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP[ built-in locations](https://www.valueserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).

**Google Maps**
Formatting the `location` parameter for Google Maps requests

To get Google Maps results based on latitude and longitude coordinates you should specify your `location` parameter in the form `location=lat:43.437677,lon:-3.8392765,zoom:15` where `43.437677` is your latitude value, `-3.8392765` is your longitude value and `15` is your zoom value.

Delimit each section with a comma `,`, and delimit each name/value pair with a colon `:`.

Valid `zoom` values are between `3` (maximum zoom-out) and `21` (maximum zoom-in).

Google can return results outside the bounds of the zoom level in some instances. If this behaviour is not desirable then specifying `strict:true` in the `location` parameter will only return results within the current zoom level. Eg. `lat:39.58467741051493,lon:-0.6752313712718961,zoom:15,strict:true`
        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        num: Note the maximum number of results per page Google allow, for Places results, is `20`.

Determines the number of results to show per `page`. Use in combination with the `page` parameter to implement pagination. 

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com`.
        location_auto: If the `location` field is set to a VALUE SERP [built-in location](https://www.valueserp.com/docs/locations-api) from the [Locations API](https://www.valueserp.com/docs/locations-api), and `location_auto` is set to `true` (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.
        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {}
    if page:
        querystring['page'] = page
    if hl:
        querystring['hl'] = hl
    if location:
        querystring['location'] = location
    if gl:
        querystring['gl'] = gl
    if num:
        querystring['num'] = num
    if google_domain:
        querystring['google_domain'] = google_domain
    if location_auto:
        querystring['location_auto'] = location_auto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_video(q: str, type: str, page: int=None, num: int=None, time_period_max: str=None, safe: str=None, nfpr: int=None, time_period_min: str=None, cr: str=None, time_period: str=None, hl: str=None, uule: str=None, gl: str=None, lr: str=None, google_domain: str=None, location_auto: bool=None, location: str=None, filter: int=None, tbs: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Video Parameters are applicable when making a request with search_type=videos to retrieve video results for a given search term. The search term is specified in the q parameter and the optional location parameter can be used to geo-locate the videos request (locations can be retrieved via the Locations API)."
    q: The keyword you want to use to perform the search.
        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination
        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.
        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.valueserp.com/docs/search-api/reference/google-cr-countries).

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_min` or `time_period_max` parameters to define the custom time period.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. VALUE SERP automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.valueserp.com/docs/search-api/reference/google-lr-languages).

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com`.
        location_auto: If the `location` field is set to a VALUE SERP [built-in location](https://www.valueserp.com/docs/locations-api) from the [Locations API](https://www.valueserp.com/docs/locations-api), and `location_auto` is set to `true` (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.
        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP[ built-in locations](https://www.valueserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).
        filter: Determines if the filters for `Similar Results` and `Omitted Results` are on or off. Can be set to `1 ` (default) to enable these filters, or `0` to disable these filters.

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'q': q, 'type': type, }
    if page:
        querystring['page'] = page
    if num:
        querystring['num'] = num
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if safe:
        querystring['safe'] = safe
    if nfpr:
        querystring['nfpr'] = nfpr
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if cr:
        querystring['cr'] = cr
    if time_period:
        querystring['time_period'] = time_period
    if hl:
        querystring['hl'] = hl
    if uule:
        querystring['uule'] = uule
    if gl:
        querystring['gl'] = gl
    if lr:
        querystring['lr'] = lr
    if google_domain:
        querystring['google_domain'] = google_domain
    if location_auto:
        querystring['location_auto'] = location_auto
    if location:
        querystring['location'] = location
    if filter:
        querystring['filter'] = filter
    if tbs:
        querystring['tbs'] = tbs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_image(q: str, type: str, images_usage: str=None, images_color: str=None, time_period_min: str=None, safe: str=None, images_page: str=None, gl: str=None, lr: str=None, hl: str=None, location_auto: bool=None, uule: str=None, google_domain: str=None, images_type: str=None, images_size: str=None, time_period_max: str=None, location: str=None, tbs: str=None, cr: str=None, time_period: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Image Parameters are applicable when making a request with search_type=images to retrieve image results for a given search term. The search term is specified in the q parameter and the optional location parameter can be used to geo-locate the image request (locations can be retrieved via the Locations API)."
    q: The keyword you want to use to perform the search.
        images_usage: Allows you to specify the usage rights of the images returned when `search_type=images`. Valid values are r`euse_with_modification`, `reuse`, `non_commercial_reuse_with_modification` or `non_commercial_reuse`.

        images_color: Allows you to set the color of the images returned when `search_type=images`. Valid values are `any`, `black_and_white`, `transparent`, `red`, `orange`, `yellow`, `green`, `teal`, `blue`, `purple`, `pink`, `white`, `gray`, `black` or `brown`.

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        images_page: Determines the page of results to get when `search_type=images`. For images Google will return 100 results per page (a Google-imposed limit). Use `images_page` to specify the page of results to retrieve, for example is `images_page=2` the API will return images 100-199.

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.valueserp.com/docs/search-api/reference/google-lr-languages).

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location_auto: If the `location` field is set to a VALUE SERP [built-in location](https://www.valueserp.com/docs/locations-api) from the [Locations API](https://www.valueserp.com/docs/locations-api), and `location_auto` is set to `true` (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.
        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. VALUE SERP automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com`.
        images_type: Allows you to set the size of the images returned when `search_type=images`. Allows you to set the type of the images returned when `search_type=images`. Valid values are `clipart`, `line_drawing`, or `gif`.

        images_size: Allows you to set the size of the images returned when `search_type=images`. Valid values are `large`, `medium`, or `icon`.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP[ built-in locations](https://www.valueserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).
        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.valueserp.com/docs/search-api/reference/google-cr-countries).

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_min` or `time_period_max` parameters to define the custom time period.

        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'q': q, 'type': type, }
    if images_usage:
        querystring['images_usage'] = images_usage
    if images_color:
        querystring['images_color'] = images_color
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if safe:
        querystring['safe'] = safe
    if images_page:
        querystring['images_page'] = images_page
    if gl:
        querystring['gl'] = gl
    if lr:
        querystring['lr'] = lr
    if hl:
        querystring['hl'] = hl
    if location_auto:
        querystring['location_auto'] = location_auto
    if uule:
        querystring['uule'] = uule
    if google_domain:
        querystring['google_domain'] = google_domain
    if images_type:
        querystring['images_type'] = images_type
    if images_size:
        querystring['images_size'] = images_size
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if location:
        querystring['location'] = location
    if tbs:
        querystring['tbs'] = tbs
    if cr:
        querystring['cr'] = cr
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_news(search_type: str, tbs: str=None, uule: str=None, sort_by: str=None, exclude_if_modified: bool=None, num: int=None, time_period: str=None, show_duplicates: bool=None, nfpr: int=None, time_period_max: str=None, time_period_min: str=None, location_auto: bool=None, lr: str=None, filter: int=None, hl: str=None, location: str=None, cr: str=None, google_domain: str=None, safe: str=None, gl: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google News Parameters are applicable when making a request with search_type=news to retrieve news results for a given search term. The search term is specified in the q parameter and the optional location parameter can be used to geo-locate the news request (locations can be retrieved via the Locations API).
		
		You can use the time_period request parameter to refine your news search to specific time periods."
    tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. VALUE SERP automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        sort_by: Determines how results are sorted, valid values are `relevance` or `date`. Note that when `sort_by=date` the `show_duplicates` parameter can be used.

        exclude_if_modified: Determines whether the `news_results` returned when Google modifies the original query. Set to `true` to exclude `news_results` when Google modifies the original query or `false` (the default) to include `news_results` when Google modifies the original query.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination. 

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specify one or both of the `time_period_min` or `time_period_max` parameters to define the custom time period.

        show_duplicates: Determines whether duplicates are shown in the news results. Must be used in conjunction with the sort_by parameter where `sort_by` is set to date. Valid values are `true` or `false`. Defaults to `false`.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0 `(default) to include them.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        location_auto: If the `location` field is set to a VALUE SERP [built-in location](https://www.valueserp.com/docs/locations-api) from the [Locations API](https://www.valueserp.com/docs/locations-api), and `location_auto` is set to `true` (default) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location. Valid values are `true` (default) to enable this behaviour or `false` to disable.
        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.valueserp.com/docs/search-api/reference/google-lr-languages).

        filter: Determines if the filters for` Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.valueserp.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the VALUE SERP[ built-in locations](https://www.valueserp.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto` parameter).
        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.valueserp.com/docs/search-api/reference/google-cr-countries).

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.valueserp.com/docs/search-api/reference/google-domains). Defaults to `google.com`.
        safe: Determines whether `Safe Search` is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        gl: The `gl` parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](https://www.valueserp.com/docs/search-api/reference/google-countries). Defaults to `us`.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the num parameter to implement pagination.
        
    """
    url = f"https://value-serp.p.rapidapi.com/search"
    querystring = {'search_type': search_type, }
    if tbs:
        querystring['tbs'] = tbs
    if uule:
        querystring['uule'] = uule
    if sort_by:
        querystring['sort_by'] = sort_by
    if exclude_if_modified:
        querystring['exclude_if_modified'] = exclude_if_modified
    if num:
        querystring['num'] = num
    if time_period:
        querystring['time_period'] = time_period
    if show_duplicates:
        querystring['show_duplicates'] = show_duplicates
    if nfpr:
        querystring['nfpr'] = nfpr
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if location_auto:
        querystring['location_auto'] = location_auto
    if lr:
        querystring['lr'] = lr
    if filter:
        querystring['filter'] = filter
    if hl:
        querystring['hl'] = hl
    if location:
        querystring['location'] = location
    if cr:
        querystring['cr'] = cr
    if google_domain:
        querystring['google_domain'] = google_domain
    if safe:
        querystring['safe'] = safe
    if gl:
        querystring['gl'] = gl
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "value-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

