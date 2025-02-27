import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def yandex_search(engine: str, q: str, page: str=None, yandex_domain: str=None, yandex_language: str=None, yandex_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Yandex Search Parameters are applicable when making a request with `engine=yandex` to retrieve Yandex search results for a given search term. The search term is specified in the `q` parameter and the optional `yandex_domain` parameter can be used to specify the Yandex domain used for the request. Supported Yandex domains are `yandex.com` and `yandex.ru`.
		
		Setting the location of Yandex requests
		The `yandex_location` parameter can be used to localize your Yandex request. The `yandex_location` parameter is only available on the `yandex.ru` domain - i.e. when your request has the `yandex_domain=yandex.ru` request parameter set.
		
		View the full list of supported `yandex_location` values [here](https://www.serpwow.com/docs/search-api/reference/yandex-locations)"
    page: Determines the page of results to return, defaults to `1`.

        yandex_domain: The Yandex domain to retrieve search results from. Supported Yandex domains are `yandex.com` and `yandex.ru`. Defaults to `yandex.com`.

        yandex_language: The `yandex_language` parameter determines the Yandex UI language to return results. View the full list of supported `yandex_language` values [here](https://www.serpwow.com/docs/search-api/reference/yandex-languages).

        yandex_location: The `yandex_location` parameter determines the Yandex-defined location to show results from. This is typically the country or location in which the user is making requests. For example `yandex_location=177` for Berlin, Germany.

View the full list of supported `yandex_location` values [here](https://www.serpwow.com/docs/search-api/reference/yandex-locations).

Note the `yandex_location` parameter only functions when `yandex_domain=yandex.ru`. It cannot be used with any other `yandex_domain`.
        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'q': q, }
    if page:
        querystring['page'] = page
    if yandex_domain:
        querystring['yandex_domain'] = yandex_domain
    if yandex_language:
        querystring['yandex_language'] = yandex_language
    if yandex_location:
        querystring['yandex_location'] = yandex_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations(q: str, parent_id: str=None, type: str=None, limit: str=None, country_code: str=None, page: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The SerpWow Locations API allows you to search for SerpWow supported Google search locations. You can supply the `full_name` returned by the Locations API as the `location` parameter in a Search API query to retrieve search results localized to that location."
    q: The query to use to search for locations, i.e. new york or mumbai.

i.e. `London`
        parent_id: Limits the search to child locations of the given `parent_id`. Useful for retrieving all sub-locations of a given location (all cities within a country, for example).

        type: Limits the results to a specific type of location. Valid values are: `city`, `country`, `county`, `dma_region`, `municipality`, `neighborhood`, `state`, `postal_code` or `province`. Multiple location types should be presented as a comma separated list (i.e.`type=state,city` to include Location results that are of type `state` or `city`).

        limit: Limits the number of search results returned per page. Defaults to `10`. The maximum number of location results returned per page is `100`. Use in conjunction with the the `page` parameter to implement pagination.

        country_code: Limits the results to locations in specific countries. See [supported countries](https://www.serpwow.com/docs/search-api/reference/google-countries) for a full list of all supported values. Multiple countries should be presented as a comma separated list (i.e. `country_code=de,fr,us` for Germany, France and the United States).

        page: Specifies the page of results to retrieve from `0` (first page) to a maximum page number of `100` (SerpWow does not support paging beyond the 100th page). Defaults to `0`. Use in conjunction with the limit parameter to limit the number of results returned per page.

        is_id: The `id` of a specific location (location ids are returned with all results from the Locations API).

        
    """
    url = f"https://serpwow.p.rapidapi.com/locations"
    querystring = {'q': q, }
    if parent_id:
        querystring['parent_id'] = parent_id
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    if country_code:
        querystring['country_code'] = country_code
    if page:
        querystring['page'] = page
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def naver_news(q: str, search_type: str, engine: str, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Naver News Parameters are applicable when making a request with `engine=naver` and `search_type=news` to retrieve Naver News results for a given search term. The search term is specified in the `q` parameter."
    q: Should be set to `engine=naver`.

        engine: The keyword you want to use to perform the search.

        page: Determines the page of results to return, defaults to `1`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'search_type': search_type, 'engine': engine, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def naver_search(q: str, engine: str, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Naver Search Parameters are applicable when making a request with `engine=naver` to retrieve Naver search results for a given search term. The search term is specified in the `q` parameter."
    q: Should be set to `engine=naver`.

        engine: The keyword you want to use to perform the search.

        page: Determines the page of results to return, defaults to `1`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'engine': engine, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baidu_search(engine: str, q: str, page: str=None, query_type: str=None, results_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Baidu Search Parameters are applicable when making a request with `engine=baidu` to retrieve Baidu search results for a given search term. The search term is specified in the `q` parameter."
    engine: Should be set to `engine=baidu`.

        q: The keyword you want to use to perform the search.

        page: Determines the page of results to return, defaults to `1`.

        query_type: Determines where Baidu searches for your query, can be set to `page_title` (to have Baidu search within the page title - similar to Google's `intitle:` query parameter) or web_address (to have Baidu search within the sites `web address` - similar to Google's `inurl:` query parameter).

        results_language: Determines whether Baidu limits the results to those in specific languages, can be set to `all` (to remove any limit), `simplified_chinese` or `traditional_chinese`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'q': q, }
    if page:
        querystring['page'] = page
    if query_type:
        querystring['query_type'] = query_type
    if results_language:
        querystring['results_language'] = results_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yahoo_news(engine: str, search_type: str, q: str, page: str=None, country_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Yahoo News Parameters are applicable when making a request with `engine=yahoo` and `search_type=news` to retrieve Yahoo News results for a given search term. The search term is specified in the `q` parameter and the optional `country_code` parameter can be used to specify Yahoo country to use."
    engine: Should be set to `engine=yahoo`.

        q: The keyword you want to use to perform the search.

        page: Determines the page of results to return, defaults to `1`.

        country_code: The country_code parameter determines the country to show results from. View the full list of supported Yahoo `country_code` values view the [Yahoo Country Codes reference](https://www.serpwow.com/docs/search-api/reference/yahoo-country-codes).

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'search_type': search_type, 'q': q, }
    if page:
        querystring['page'] = page
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yahoo_search(q: str, engine: str, page: str=None, country_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Yahoo Search Parameters are applicable when making a request with `engine=yahoo` to retrieve Yahoo search results for a given search term. The search term is specified in the `q` parameter."
    q: The keyword you want to use to perform the search.

        engine: Should be set to `engine=yahoo`.

        page: Determines the page of results to return, defaults to `1`.

        country_code: The country_code parameter determines the country to show results from. View the full list of supported Yahoo `country_code` values view the [Yahoo Country Codes reference](https://www.serpwow.com/docs/search-api/reference/yahoo-country-codes).

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'engine': engine, }
    if page:
        querystring['page'] = page
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bing_search(q: str, engine: str, page: int=None, include_answer_box: str=None, market_code: str=None, bing_language: str=None, country_code: str=None, num: str=None, safe: str=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Bing Search Parameters are applicable when making a request with `engine=bing` to retrieve Bing search results for a given search term. The search term is specified in the `q` parameter. The location your search is run from is determined by the `location` parameter, which can be populated with a `full_name` value from the Locations API.
		
		**Setting Location for Bing requests**
		The recommended way of setting the `location` of your Bing request is to use the `location` parameter. The `location` parameter can be populated with a `full_name` value returned by the Locations API.
		
		You may also use either the `market_code` or `country_code` parameters to set the location of your request, but the recommended approach is to use the location parameter as this sets all the required prerequisites to achieve a correctly geo-located result."
    q: The keyword you want to use to perform the search.
        engine: Should be set to `engine=bing`.
        page: Determines the page of results to return, defaults to `1`.

        include_answer_box: Determines whether to include the answer box (sometimes called the "featured snippet") in the `organic_results` array and treat it as the first result. This may be desirable if you treat the result Bing displayed in the `answer_box` as the first organic result.

        market_code: 
The `market_code` parameter determines the Bing-defined market to show results from. This is typically the country in which the user is making requests.

View the full list of supported Bing `market_code` values [here](https://www.serpwow.com/docs/search-api/reference/bing-market-codes).

**Note** the `market_code` parameter cannot be used in combination with the `country_code` parameter.
        bing_language: The `bing_language` parameter determines the Bing UI language to return results. View the full list of supported `bing_language` values [here](https://www.serpwow.com/docs/search-api/reference/bing-languages). Defaults to `en`.

        country_code: The `country_code` parameter determines the country to show results from (if `market_code` is not specified).

View the full list of supported Bing `country_code` values [here](https://www.serpwow.com/docs/search-api/reference/bing-country-codes).

**Note** it is recommended that the `market_code` parameter be used instead of `country_code`. The `country_code` parameter cannot be used in combination with the `market_code` parameter.
        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination.

**Note** `num` must be set to `50` or less (this is a Bing-imposed limitation of not returning more than 50 results per page).
        safe: Determines whether `Safe Search` is enabled for the results. Can be set to strict to enable `Safe Search`, `moderate` for just images and videos or `off` to disable Safe Search.

        location: Determines the geographic location in which the query is executed. You must enter a `full_name` value from the [SerpWow Locations API](https://www.serpwow.com/docs/locations-api/overview), for example `location=Richmond+County,New+York,United+States` (where `Richmond+County,New+York,United+States` is the `full_name` as-returned by the Locations API).

**Note** the `location` parameter is the recommended way of setting the location of your Bing request.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'engine': engine, }
    if page:
        querystring['page'] = page
    if include_answer_box:
        querystring['include_answer_box'] = include_answer_box
    if market_code:
        querystring['market_code'] = market_code
    if bing_language:
        querystring['bing_language'] = bing_language
    if country_code:
        querystring['country_code'] = country_code
    if num:
        querystring['num'] = num
    if safe:
        querystring['safe'] = safe
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bing_image(type: str, engine: str, q: str, location: str=None, country_code: str=None, bing_language: str=None, market_code: str='images', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Bing Image Parameters are applicable when making a request with `engine=bing` and `search_type=images` to retrieve image results for a given search term. The search term is specified in the `q` parameter and the optional `location` parameter can be used to geo-locate the image request (locations can be retrieved via the Locations API)."
    engine: Should be set to `engine=bing`.
        q: The keyword you want to use to perform the search.
        location: Determines the geographic location in which the query is executed. You must enter a `full_name` value from the [SerpWow Locations API](https://www.serpwow.com/docs/locations-api/overview), for example `location=Richmond+County,New+York,United+States` (where `Richmond+County,New+York,United+States` is the `full_name` as-returned by the Locations API).

**Note** the `location` parameter is the recommended way of setting the location of your Bing request.

        country_code: The `country_code` parameter determines the country to show results from (if `market_code` is not specified).

View the full list of supported Bing `country_code` values [here](https://www.serpwow.com/docs/search-api/reference/bing-country-codes).

**Note** it is recommended that the `market_code` parameter be used instead of `country_code`. The `country_code` parameter cannot be used in combination with the `market_code` parameter.
        bing_language: The `bing_language` parameter determines the Bing UI language to return results. View the full list of supported `bing_language` values [here](https://www.serpwow.com/docs/search-api/reference/bing-languages). Defaults to `en`.

        market_code: 
The `market_code` parameter determines the Bing-defined market to show results from. This is typically the country in which the user is making requests.

View the full list of supported Bing `market_code` values [here](https://www.serpwow.com/docs/search-api/reference/bing-market-codes).

**Note** the `market_code` parameter cannot be used in combination with the `country_code` parameter.
        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'type': type, 'engine': engine, 'q': q, }
    if location:
        querystring['location'] = location
    if country_code:
        querystring['country_code'] = country_code
    if bing_language:
        querystring['bing_language'] = bing_language
    if market_code:
        querystring['market_code'] = market_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bing_news(engine: str, q: str, type: str, bing_language: str=None, location: str=None, country_code: str=None, market_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Bing News Parameters are applicable when making a request with `engine=bing` and `search_type=news` to retrieve Bing News results for a given search term. The search term is specified in the `q` parameter and the optional `country_code` parameter can be used to specify Bing country to use."
    engine: Should be set to `engine=bing`.
        q: The keyword you want to use to perform the search.
        bing_language: The `bing_language` parameter determines the Bing UI language to return results. View the full list of supported `bing_language` values [here](https://www.serpwow.com/docs/search-api/reference/bing-languages). Defaults to `en`.

        location: Determines the geographic location in which the query is executed. You must enter a `full_name` value from the [SerpWow Locations API](https://www.serpwow.com/docs/locations-api/overview), for example `location=Richmond+County,New+York,United+States` (where `Richmond+County,New+York,United+States` is the `full_name` as-returned by the Locations API).

**Note** the `location` parameter is the recommended way of setting the location of your Bing request.

        country_code: The `country_code` parameter determines the country to show results from (if `market_code` is not specified).

View the full list of supported Bing `country_code` values [here](https://www.serpwow.com/docs/search-api/reference/bing-country-codes).

**Note** it is recommended that the `market_code` parameter be used instead of `country_code`. The `country_code` parameter cannot be used in combination with the `market_code` parameter.
        market_code: 
The `market_code` parameter determines the Bing-defined market to show results from. This is typically the country in which the user is making requests.

View the full list of supported Bing `market_code` values [here](https://www.serpwow.com/docs/search-api/reference/bing-market-codes).

**Note** the `market_code` parameter cannot be used in combination with the `country_code` parameter.
        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'q': q, 'type': type, }
    if bing_language:
        querystring['bing_language'] = bing_language
    if location:
        querystring['location'] = location
    if country_code:
        querystring['country_code'] = country_code
    if market_code:
        querystring['market_code'] = market_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_reviews(search_type: str, engine: str, product_type: str, product_id: str, google_domain: str=None, location: str=None, hl: str=None, location_auto: bool=None, uule: str=None, gl: str=None, next_page_token: str=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Reviews Parameters are applicable when making a request with `search_type=product` and `product_type=reviews` to retrieve product reviews results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a `location` parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		`Products & Location`
		Google Product pages are highly location-sensitive so it is important that you specify a location that matches the location that was used to retrieve the `product_id` in the original Google Shopping search request.
		
		**Place Reviews Pagination**
		Product Reviews results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `product_type=reviews` result will return a `next_page_token` in its' `product_results.pagination` object. This ![](next_page_token) can be passed in to the `next_page_token` request parameter to retrieve the next page of Product Reviews results."
    google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        next_page_token: Product Reviews results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `product_type=reviews` result will return a `next_page_token` in its' `product_results.pagination` object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of Product Reviews results.

        sort_by: Sets the sort ordering of the product reviews returned. Valid values are:
`relevance`
Sort product reviews results by relevance, the default.
`date`
Sort product reviews results by date, most recent first.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'product_type': product_type, 'product_id': product_id, }
    if google_domain:
        querystring['google_domain'] = google_domain
    if location:
        querystring['location'] = location
    if hl:
        querystring['hl'] = hl
    if location_auto:
        querystring['location_auto'] = location_auto
    if uule:
        querystring['uule'] = uule
    if gl:
        querystring['gl'] = gl
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_specifications(search_type: str, engine: str, product_id: str, product_type: str, location_auto: bool=None, gl: str=None, google_domain: str=None, hl: str=None, location: str=None, uule: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Specifications Parameters are applicable when making a request with `search_type=product` and `product_type=specifications` to retrieve product specifications results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a `location` parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a `location` that matches the `location` that was used to retrieve the product_id in the original Google Shopping search request."
    location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        page: Determines the page of results to return, defaults to `1`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'product_id': product_id, 'product_type': product_type, }
    if location_auto:
        querystring['location_auto'] = location_auto
    if gl:
        querystring['gl'] = gl
    if google_domain:
        querystring['google_domain'] = google_domain
    if hl:
        querystring['hl'] = hl
    if location:
        querystring['location'] = location
    if uule:
        querystring['uule'] = uule
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_place_and_maps_products_parameters(search_type: str, engine: str, data_cid: str, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Products Parameters are applicable when making a request with `search_type=place_products` to retrieve place products for a Place. The Place is specified in the `data_cid` parameter, `data_cid` values are returned from `search_type=places` Places requests.
		
		**`data_cid` IDs used in `place_products` requests**
		Note that `place_products` requests differ from `place_reviews` (docs) and `place_photos` (docs) requests in that it required a `data_cid` instead of a `data_id` request parameter.
		
		Both `data_cid` and `data_id` IDs are returned in Places requests requests."
    search_type: Should be set to `search_type=place_details`.
        engine: Should be set to `engine=google`.
        data_cid: The `data_cid` of the Place to retrieve place products for `data_cid` values are returned in Places requests.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'data_cid': data_cid, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_online_sellers(product_type: str, search_type: str, engine: str, product_id: str, page: int=None, product_condition_new: bool=None, product_condition_used: bool=None, sort_by: str=None, location: str=None, location_auto: bool=None, uule: str=None, hl: str=None, gl: str=None, google_domain: str=None, product_free_shipping: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Online Sellers Parameters are applicable when making a request with `search_type=product` and `product_type=sellers_online` to retrieve product online sellers results for a given product ID. The product ID is specified in the `product_id` parameter and you should also specify a `location` parameter to geo-locate the request (locations can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a `location` that matches the location that was used to retrieve the `product_id` in the original Google Shopping search request."
    page: Determines the page of results to return, defaults to `1`.

        product_condition_new: Determines whether to filter to only new (non-used) products. Valid values are `true` or `false`.

        product_condition_used: Determines whether to filter to only used (non-new) products. Valid values are `true` or `false`.

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
        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        product_free_shipping: Determines whether to filter to only products with free shipping. Valid values are `true` or `false`.
        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'product_type': product_type, 'search_type': search_type, 'engine': engine, 'product_id': product_id, }
    if page:
        querystring['page'] = page
    if product_condition_new:
        querystring['product_condition_new'] = product_condition_new
    if product_condition_used:
        querystring['product_condition_used'] = product_condition_used
    if sort_by:
        querystring['sort_by'] = sort_by
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
    if google_domain:
        querystring['google_domain'] = google_domain
    if product_free_shipping:
        querystring['product_free_shipping'] = product_free_shipping
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_product_search(engine: str, product_id: str, search_type: str, google_domain: str=None, uule: str=None, location_auto: bool=None, gl: str=None, location: str=None, hl: str=None, production_condition_used: bool=None, product_free_shipping: bool=None, product_condition_new: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Product Parameters are applicable when making a request with `search_type=product` to retrieve product results for a given product ID. The product ID is specified in the product_id parameter and you should also specify a location parameter to geo-locate the request (`locations` can be retrieved via the Locations API).
		
		Google Product IDs are returned by Google Shopping search requests.
		
		**Products & Location**
		Google Product pages are highly location-sensitive so it is important that you specify a `location` that matches the location that was used to retrieve the `product_id` in the original Google Shopping search request."
    google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        production_condition_used: Determines whether to filter to only used (non-new) products. Valid values are `true` or `false`.

        product_free_shipping: Determines whether to filter to only products with free shipping. Valid values are `true` or `false`.

        product_condition_new: Determines whether to filter to only new (non-used) products. Valid values are `true` or `false`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'product_id': product_id, 'search_type': search_type, }
    if google_domain:
        querystring['google_domain'] = google_domain
    if uule:
        querystring['uule'] = uule
    if location_auto:
        querystring['location_auto'] = location_auto
    if gl:
        querystring['gl'] = gl
    if location:
        querystring['location'] = location
    if hl:
        querystring['hl'] = hl
    if production_condition_used:
        querystring['production_condition_used'] = production_condition_used
    if product_free_shipping:
        querystring['product_free_shipping'] = product_free_shipping
    if product_condition_new:
        querystring['product_condition_new'] = product_condition_new
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_shopping_search(q: str, search_type: str, shopping_condition: str=None, shopping_price_min: str=None, shopping_price_max: str=None, shopping_merchants: str=None, location_auto: bool=None, gl: str=None, page: int=None, uule: str=None, shopping_buy_on_google: bool=None, num: int=None, google_domain: str=None, hl: str=None, location: str=None, shopping_filter: str=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Shopping Parameters are applicable when making a request with `search_type=shopping` to retrieve Shopping results for a given search term. The search term is specified in the `q `parameter and the optional location parameter can be used to geo-locate the shopping request (`locations` can be retrieved via the [Locations API](https://www.serpwow.com/docs/locations-api/overview))."
    q: The keyword you want to use to perform the search.
        shopping_condition: The condition of products returned. Can be set to `new` or `used`.

        shopping_price_min: The minimum price of products. For example `shopping_price_min=4.99`

        shopping_price_max: The maximum price of products returned. For example `shopping_price_max=29.99`

        shopping_merchants: A comma-separated list of merchant IDs to retrieve shopping results. Merchant ID's can be found in the `merchagg:` section of any Google Shopping URL.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        shopping_buy_on_google: Determines whether the \"Buy on Google\" option is selected when running a `search_type=shopping` search. Valid values are `true` or `false`.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        shopping_filter: A shopping filter (i.e. \"Brand\") to filter the results to. Shopping filter values are returned in the `filters` property of the shopping response.

        sort_by: Sets the sort ordering of the shopping results returned. Valid values are:

`relevance`
Sort shopping results by relevance to the search term supplied in the q parameter, the default.

`price_low_to_high`
Sort shopping results by lowest to highest price.

`price_high_to_low`
Sort shopping results by highest to lowest price.

`review_score`
Sort shopping results by review score, highest review score first
        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'search_type': search_type, }
    if shopping_condition:
        querystring['shopping_condition'] = shopping_condition
    if shopping_price_min:
        querystring['shopping_price_min'] = shopping_price_min
    if shopping_price_max:
        querystring['shopping_price_max'] = shopping_price_max
    if shopping_merchants:
        querystring['shopping_merchants'] = shopping_merchants
    if location_auto:
        querystring['location_auto'] = location_auto
    if gl:
        querystring['gl'] = gl
    if page:
        querystring['page'] = page
    if uule:
        querystring['uule'] = uule
    if shopping_buy_on_google:
        querystring['shopping_buy_on_google'] = shopping_buy_on_google
    if num:
        querystring['num'] = num
    if google_domain:
        querystring['google_domain'] = google_domain
    if hl:
        querystring['hl'] = hl
    if location:
        querystring['location'] = location
    if shopping_filter:
        querystring['shopping_filter'] = shopping_filter
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_place_and_maps_posts(engine: str, search_type: str, data_id: str, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Posts Parameters are applicable when making a request with `search_type=place_posts` to retrieve place posts for a Place. The Place is specified in the `data_id` parameter, `data_id` values are returned from `search_type=places` Places requests."
    engine: Should be set to `engine=google`.
        search_type: Should be set to `search_type=place_details`.
        data_id: The `data_id` of the Place to retrieve place posts for `data_id` values are returned in Places requests.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'search_type': search_type, 'data_id': data_id, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_place_and_maps_reviews(engine: str, search_type: str, data_id: str, next_page_token: str=None, hl: str=None, topic_id: str=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Reviews Parameters are applicable when making a request with `search_type=place_reviews` to retrieve place reviews for a Place. The Place is specified in the `data_id` parameter, `data_id` values are returned from `search_type=places` Places requests.
		
		**Place Review Topic IDs**
		If shown, a `search_type=place_reviews` will return a topics array containing review topics. You can pass a place reviews topic ID in to the optional `topic_id` parameter to refine your place reviews request to just reviews related to that topic.
		
		**Place Reviews Pagination**
		Place Review results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `search_type=place_reviews` result will return a `next_page_token` in its' pagination object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of place reviews results."
    engine: Should be set to `engine=google`.
        search_type: Should be set to `search_type=place_details`.
        data_id: The data_id of the Place to retrieve place details for. `data_id` values are returned in Places requests.

Note that either a `data_id` or `data_cid` identifier must be supplied.
        next_page_token: Place Photo results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `search_type=place_photos` result will return a `next_page_token` in its' `pagination` object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of place photo results.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        topic_id: An optional topic ID to limit the returned reviews to just those related to the given topic. topic_id values, if shown for the given `data_id`, are returned in the topics array in all `search_type=place_reviews` results.

        sort_by: Determines how place reviews are sorted, valid values are `relevance`, `date`, `highest_rating` and `lowest_rating`. Defaults to `relevance`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'search_type': search_type, 'data_id': data_id, }
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    if hl:
        querystring['hl'] = hl
    if topic_id:
        querystring['topic_id'] = topic_id
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_place_and_maps_photos(search_type: str, engine: str, data_id: str, hl: str=None, next_page_token: str=None, category_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Photos Parameters are applicable when making a request with `search_type=place_photos` to retrieve place photos for a Place. The Place is specified in the `data_id` parameter, `data_id` values are returned from `search_type=places` Places requests.
		
		**Place Photo Category IDs**
		If shown, a `search_type=place_photos` will return a `categories` array containing photo categories. You can pass a place photos category ID in to the optional `category_id` parameter to refine your place photos request to just photos belonging to that category.
		
		**Place Photos Pagination**
		Place Photo results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `search_type=place_photos` result will return a `next_page_token` in its' pagination object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of place photo results."
    search_type: Should be set to `search_type=place_details`.
        engine: Should be set to `engine=google`.
        data_id: The data_id of the Place to retrieve place details for. `data_id` values are returned in Places requests.

Note that either a `data_id` or `data_cid` identifier must be supplied.
        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        next_page_token: Place Photo results do not contain a traditional pagination section where a specific page of results is uniquely addressable. Instead, each `search_type=place_photos` result will return a `next_page_token` in its' `pagination` object. This `next_page_token` can be passed in to the `next_page_token` request parameter to retrieve the next page of place photo results.

        category_id: An optional category ID to limit the returned photos to just those belonging to the given category. `category_id` values, if shown for the given `data_id`, are returned in the categories array in all `search_type=place_photos` results.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'data_id': data_id, }
    if hl:
        querystring['hl'] = hl
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    if category_id:
        querystring['category_id'] = category_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_place_and_maps_details(engine: str, search_type: str, data_cid: str, data_id: str, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Place Details Parameters are applicable when making a request with `search_type=place_details` to retrieve place details for a Place. The Place is specified in the `data_id` parameter, `data_id` values are returned from `search_type=places` Places requests.
		
		If the `data_id` is not available you may also request a `place_details` request using a `data_cid` (a standardised Google Place identifier you may already have access to). However, `data_id` is the recommended identifier to use."
    engine: Should be set to `engine=google`.
        search_type: Should be set to `search_type=place_details`.
        data_cid: The `data_cid` (a standard Google Place identifier) of the Place to retrieve place details for. `data_cid` values are returned in Places requests.

Note that either a `data_id` or `data_cid` identifier must be supplied.

        data_id: The data_id of the Place to retrieve place details for. `data_id` values are returned in Places requests.

Note that either a `data_id` or `data_cid` identifier must be supplied.
        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'engine': engine, 'search_type': search_type, 'data_cid': data_cid, 'data_id': data_id, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_places_and_maps_search(search_type: str, engine: str, q: str, num: int=None, page: int=None, google_domain: str=None, gl: str=None, nfpr: int=None, location_auto: bool=None, location: str=None, uule: str=None, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SerpWow parses Google Local & Maps results (i.e. local business listings) when the search_type parameter is set to search_type=places. When this parameter is set SerpWow provides a places_results array in the result JSON that contains the [Places or Maps results](https://www.serpwow.com/docs/search-api/searches/google/places).
		
		**Google Local vs. Google Maps**
		`search_type=places` requests can be executed in two modes to retrieve data from a [Google Local results page](https://www.google.com/search?q=pizza&gl=us&hl=en&uule=w+CAIQICIgTWFuaGF0dGFuLE5ldyBZb3JrLFVuaXRlZCBTdGF0ZXM&tbm=lcl) or from a [Google Maps results page](https://www.google.com/maps/search/pizza/@43.437677,-3.8392765,15z).
		
		You set the Location of the `search_type=places` request using the `location` request parameter and this can be expressed as either a text location name from the [Locations API](https://www.serpwow.com/docs/search-api/searches/google/places#:~:text=SerpWow%20parses%20Google,Maps%20result).) (which will result in a [Google Local result](https://www.google.com/search?q=pizza&gl=us&hl=en&uule=w+CAIQICIgTWFuaGF0dGFuLE5ldyBZb3JrLFVuaXRlZCBTdGF0ZXM&tbm=lcl)), or as a latitude, longitude and zoom level (which will result in a [Google Maps result](https://www.google.com/maps/search/pizza/@43.437677,-3.8392765,15z))."
    engine: Should be set to `engine=google`.

        q: The keyword you want to use to perform the search.
        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        location: Google Places [Local]
Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

Google Maps
If the location parameter is set to a latitude, longitude and zoom value - for example `location=lat:43.437677,lon:-3.8392765,zoom:15` then results will be returned from a [Google Maps](https://www.google.com/maps/search/pizza/@43.437677,-3.8392765,15z) page.

**Formatting the `location` parameter for Google Maps requests**
To get Google Maps results based on latitude and longitude coordinates you should specify your `location` parameter in the form `location=lat:43.437677,lon:-3.8392765,zoom:15` where `43.437677` is your latitude value, `-3.8392765` is your longitude value and `15` is your zoom value.

Delimit each section with a comma `,`, and delimit each name/value pair with a colon `:`.

Valid `zoom` values are between `3` (maximum zoom-out) and `21` (maximum zoom-in).

Google can return results outside the bounds of the zoom level in some instances. If this behaviour is not desirable then specifying `strict:true` in the `location` parameter will only return results within the current zoom level. Eg. `lat:39.58467741051493,lon:-0.6752313712718961,zoom:15,strict:true`
        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'q': q, }
    if num:
        querystring['num'] = num
    if page:
        querystring['page'] = page
    if google_domain:
        querystring['google_domain'] = google_domain
    if gl:
        querystring['gl'] = gl
    if nfpr:
        querystring['nfpr'] = nfpr
    if location_auto:
        querystring['location_auto'] = location_auto
    if location:
        querystring['location'] = location
    if uule:
        querystring['uule'] = uule
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_autocomplete(search_type: str, engine: str, q: str, autocomplete_search_index: str=None, uule: str=None, location_auto: bool=None, gl: str=None, hl: str=None, location: str=None, google_domain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Autocomplete Parameters are applicable when making a request with `search_type=autocomplete` to retrieve autocomplete results for a given search term. The keyword searched can be specified in the `q` parameter and the optional location parameter can be used to geo-locate the autocomplete request (`locations` can be retrieved via the [Locations API](https://www.serpwow.com/docs/locations-api/overview))."
    engine: Should be set to `engine=google`.

        q: The keyword you want to use to perform the search.
        autocomplete_search_index: Optional parameter to determine on which index of the string provided in the `q` param the autocomplete operation is requested. Different autocomplete results are returned by Google depending on the index within the search term the user currently has their cursor set to.
        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'q': q, }
    if autocomplete_search_index:
        querystring['autocomplete_search_index'] = autocomplete_search_index
    if uule:
        querystring['uule'] = uule
    if location_auto:
        querystring['location_auto'] = location_auto
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if location:
        querystring['location'] = location
    if google_domain:
        querystring['google_domain'] = google_domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_trends(search_type: str, engine: str, q: str, trends_type: str=None, gl: str=None, hl: str=None, time_period: str=None, trends_geo: str=None, time_period_max: str=None, time_period_min: str=None, trends_resolution: str=None, trends_category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Trends Parameters are applicable when making a request with `search_type=trends` to retrieve Google Trends results for a given search term. The search term is specified in the `q` parameter.
		
		Multiple Trends Keywords
		You can request comparative Trends results for up to 5 keywords in one request. Specify multiple trends keywords in comma-separated notation in the q parameter. For example, to replicate the request shown in the screenshot below the q parameter would be `q=pizza,burger,chips,sausages,bread`."
    engine: Should be set to `engine=google`.

        q: The keyword you want to use to perform the News search.

        trends_type: Determines the Google property to search for Trends data. Valid values are `web`, `images`, `news`, `youtube` or `shopping`.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or custom. When using `custom` you must also specifiy one or both of the `time_period_minor` `time_period_max` parameters to define the `custom` time period.

        trends_geo: A Google Trends Geo ID - typically a 2-letter country code, for example `trends_geo=DE` for Trends data for Germany. Can also specify a more specific `trends_geo` for a specific location within a country. View the full list of supported `trends_geo` values [here](https://www.serpwow.com/docs/search-api/reference/google-trends-geos).

Omit the `trends_geo` parameter entirely if you wish to run your query as \"worldwide\".
        time_period_max: Determines the maximum (i.e. 'to') time to use when time_periodis set to custom. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        time_period_min: Determines the minimum (i.e. 'from') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        trends_resolution: Determines the resolution requested for the `trends_interest_by_subregion` section of SerpWow's Google Trends results. Valid values are `country`, `city`, `dma` or `region`.

        trends_category: A Google Trends Category ID. View the full list of supported trends_category values `[here](https://www.serpwow.com/docs/search-api/reference/google-trends-categories)`.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'q': q, }
    if trends_type:
        querystring['trends_type'] = trends_type
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if time_period:
        querystring['time_period'] = time_period
    if trends_geo:
        querystring['trends_geo'] = trends_geo
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if trends_resolution:
        querystring['trends_resolution'] = trends_resolution
    if trends_category:
        querystring['trends_category'] = trends_category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_scholar(q: str, engine: str, search_type: str, page: int=None, sort_by: str=None, scisbd: int=None, scholar_year_min: str=None, lr: str=None, location: str=None, uule: str=None, location_auto: bool=None, google_domain: str=None, cr: str=None, hl: str=None, gl: str=None, scholar_patents_courts: str=None, scholar_year_max: str=None, scholar_include_citations: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Scholar Parameters are applicable when making a request with `search_type=scholar` to retrieve scholar results for a given search term. The search term is specified in the `q` parameter and the optional location parameter can be used to geo-locate the scholar request (locations can be retrieved via the [Locations API](https://www.serpwow.com/docs/locations-api/overview))."
    q: The keyword you want to use to perform the News search.

        engine: Should be set to `engine=google`.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        sort_by: Determines how results are sorted, valid values are `relevance` or `date`.

        scisbd: The Scholar scisbd parameter can be set to `1` or `2`.

        scholar_year_min: Determines the year from which results should be included when `search_type=scholar`. For example, if you set `scholar_year_min=2018` the results before 2018 will be omitted. This parameter can be combined with the `scholar_year_max` parameter.

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.serpwow.com/docs/search-api/reference/google-lr-languages).

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.serpwow.com/docs/search-api/reference/google-cr-countries).

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        scholar_patents_courts: The `scholar_patents_courts` parameter can be used either as both a way of limiting the Scholar search or as a filter (this is due to how Google implements this parameter itself).

**As a Filter** the valid values are:
`0` (default) - include patents
`1` - exclude patents

**As a way of limiting a Scholar search** the valid values are:
`4` - case law (US courts only). This will select all the State and Federal courts.

To select specific courts, see the [full list of supported Google Scholar courts](https://www.serpwow.com/docs/search-api/reference/google-scholar-courts).

For example, `scholar_patents_courts=4,33,192` `4` is the required value and should always be in the first position, `33` selects all New York courts and `192` selects Tax Court. Values must be separated by a comma.
        scholar_year_max: Determines the year from which results should be included when `search_type=scholar`. For example, if you set `scholar_year_max=2019` the results after 2019 will be omitted. This parameter can be combined with the `scholar_year_min` parameter.

        scholar_include_citations: Determines whether to include citations or not when the `search_type` parameter is set to scholar. Valid values are `true` or `false`. Defaults to `true`. Drives the following option:

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'engine': engine, 'search_type': search_type, }
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if scisbd:
        querystring['scisbd'] = scisbd
    if scholar_year_min:
        querystring['scholar_year_min'] = scholar_year_min
    if lr:
        querystring['lr'] = lr
    if location:
        querystring['location'] = location
    if uule:
        querystring['uule'] = uule
    if location_auto:
        querystring['location_auto'] = location_auto
    if google_domain:
        querystring['google_domain'] = google_domain
    if cr:
        querystring['cr'] = cr
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    if scholar_patents_courts:
        querystring['scholar_patents_courts'] = scholar_patents_courts
    if scholar_year_max:
        querystring['scholar_year_max'] = scholar_year_max
    if scholar_include_citations:
        querystring['scholar_include_citations'] = scholar_include_citations
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_video(search_type: str, engine: str, q: str, num: int=None, page: int=None, time_period_max: str=None, cr: str=None, tbs: str=None, time_period: str=None, safe: str=None, gl: str=None, google_domain: str=None, uule: str=None, location: str=None, hl: str=None, lr: str=None, location_auto: bool=None, time_period_min: str=None, filter: int=None, nfpr: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Video Parameters are applicable when making a request with `search_type=videos` to retrieve video results for a given search term. The search term is specified in the q parameter and the optional `location` parameter can be used to geo-locate the videos request (locations can be retrieved via the [Locations API)](https://www.serpwow.com/docs/locations-api/overview).You can use the `time_period` request parameter to refine your news search to specific time periods.
		
		If you wish to exclude news results where Google have modified the original query set exclude_if_modified=true in your request parameters."
    engine: Should be set to `engine=google`.

        q: The keyword you want to use to perform the News search.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY,` I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.serpwow.com/docs/search-api/reference/google-cr-countries).

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specifiy one or both of the `time_period_minor` `time_period_max` parameters to define the custom time period.

        safe: Determines whether Safe Search is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.serpwow.com/docs/search-api/reference/google-lr-languages).

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        time_period_min: Determines the minimum (i.e. 'from') time to use when time_periodis set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        filter: Determines if the filters for `Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'engine': engine, 'q': q, }
    if num:
        querystring['num'] = num
    if page:
        querystring['page'] = page
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if cr:
        querystring['cr'] = cr
    if tbs:
        querystring['tbs'] = tbs
    if time_period:
        querystring['time_period'] = time_period
    if safe:
        querystring['safe'] = safe
    if gl:
        querystring['gl'] = gl
    if google_domain:
        querystring['google_domain'] = google_domain
    if uule:
        querystring['uule'] = uule
    if location:
        querystring['location'] = location
    if hl:
        querystring['hl'] = hl
    if lr:
        querystring['lr'] = lr
    if location_auto:
        querystring['location_auto'] = location_auto
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if filter:
        querystring['filter'] = filter
    if nfpr:
        querystring['nfpr'] = nfpr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_image(search_type: str, q: str, engine: str, images_size: str=None, uule: str=None, location_auto: bool=None, gl: str=None, hl: str=None, lr: str=None, google_domain: str=None, location: str=None, images_page: str=None, images_color: str=None, images_usage: str=None, cr: str=None, time_period: str=None, time_period_max: str=None, images_type: str=None, tbs: str=None, time_period_min: str=None, safe: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Image Parameters are applicable when making a request with `search_type=images` to retrieve image results for a given search term. The search term is specified in the `q` parameter and the optional `location` parameter can be used to geo-locate the image request (locations can be retrieved via the [Locations API](https://www.serpwow.com/docs/locations-api/overview)).
		You can use the time_period request parameter to refine your news search to specific time periods.
		
		If you wish to exclude news results where Google have modified the original query set exclude_if_modified=true in your request parameters."
    q: The keyword you want to use to perform the News search.

        engine: Should be set to `engine=google`.

        images_size: Allows you to set the size of the images returned when `search_type=images`. Valid values are `large`, `medium`, or `icon`.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.serpwow.com/docs/search-api/reference/google-lr-languages).

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        images_page: Determines the page of results to get when `search_type=images`. For images Google will return 100 results per page (a Google-imposed limit). Use `images_page` to specify the page of results to retrieve, for example is `images_page=2` the API will return images 100-199.

        images_color: Allows you to set the color of the images returned when `search_type=images`. Valid values are `any`, `black_and_white`, `transparent`, `red`, `orange`, `yellow`, `green`, `teal`, `blue`, `purple`, `pink`, `white`, `gray`, `black` or `brown`.

        images_usage: Allows you to specify the usage rights of the images returned when `search_type=images`. Valid values are `reuse_with_modification`, `reuse`, `non_commercial_reuse_with_modification` or `non_commercial_reuse`.

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.serpwow.com/docs/search-api/reference/google-cr-countries).

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specifiy one or both of the `time_period_minor` `time_period_max` parameters to define the custom time period.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY,` I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        images_type: Allows you to set the size of the images returned when `search_type=images`. Allows you to set the type of the images returned when `search_type=images`. Valid values are `clipart`, `line_drawing`, or `gif`.

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        time_period_min: Determines the minimum (i.e. 'from') time to use when time_periodis set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        safe: Determines whether Safe Search is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'search_type': search_type, 'q': q, 'engine': engine, }
    if images_size:
        querystring['images_size'] = images_size
    if uule:
        querystring['uule'] = uule
    if location_auto:
        querystring['location_auto'] = location_auto
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if lr:
        querystring['lr'] = lr
    if google_domain:
        querystring['google_domain'] = google_domain
    if location:
        querystring['location'] = location
    if images_page:
        querystring['images_page'] = images_page
    if images_color:
        querystring['images_color'] = images_color
    if images_usage:
        querystring['images_usage'] = images_usage
    if cr:
        querystring['cr'] = cr
    if time_period:
        querystring['time_period'] = time_period
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if images_type:
        querystring['images_type'] = images_type
    if tbs:
        querystring['tbs'] = tbs
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if safe:
        querystring['safe'] = safe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_news(q: str, engine: str, search_type: str, num: int=None, page: int=None, nfpr: int=None, filter: str=None, location_auto: bool=None, time_period: str=None, location: str=None, lr: str=None, hl: str=None, google_domain: str=None, time_period_max: str=None, uule: str=None, cr: str=None, gl: str=None, time_period_min: str=None, tbs: str=None, safe: str=None, exclude_if_modified: str=None, show_duplicates: str=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google News Parameters are applicable when making a request with search_type=news to retrieve news results for a given search term. The search term is specified in the q parameter and the optional location parameter can be used to geo-locate the news request (locations can be retrieved via the Locations API).
		
		You can use the time_period request parameter to refine your news search to specific time periods.
		
		If you wish to exclude news results where Google have modified the original query set exclude_if_modified=true in your request parameters."
    q: The keyword you want to use to perform the News search.

        engine: Should be set to `engine=google`.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        filter: Determines if the filters for `Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specifiy one or both of the `time_period_minor` `time_period_max` parameters to define the custom time period.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.serpwow.com/docs/search-api/reference/google-lr-languages).

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY,` I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.serpwow.com/docs/search-api/reference/google-cr-countries).

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        time_period_min: Determines the minimum (i.e. 'from') time to use when time_periodis set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        safe: Determines whether Safe Search is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        exclude_if_modified: Determines whether the `news_results` returned when Google modifies the original query. Set to `true` to exclude `news_results` when Google modifies the original query or `false` (the default) to include `news_results` when Google modifies the original query.

        show_duplicates: Determines whether duplicates are shown in the news results. Must be used in conjunction with the `sort_by` parameter where sort_by is set to `date`. Valid values are `true` or `false`. Defaults to `false`.

        sort_by: Determines how results are sorted, valid values are `relevance` or `date`. Note that when `sort_by=date `the `show_duplicates` parameter can be used.

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'engine': engine, 'search_type': search_type, }
    if num:
        querystring['num'] = num
    if page:
        querystring['page'] = page
    if nfpr:
        querystring['nfpr'] = nfpr
    if filter:
        querystring['filter'] = filter
    if location_auto:
        querystring['location_auto'] = location_auto
    if time_period:
        querystring['time_period'] = time_period
    if location:
        querystring['location'] = location
    if lr:
        querystring['lr'] = lr
    if hl:
        querystring['hl'] = hl
    if google_domain:
        querystring['google_domain'] = google_domain
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if uule:
        querystring['uule'] = uule
    if cr:
        querystring['cr'] = cr
    if gl:
        querystring['gl'] = gl
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if tbs:
        querystring['tbs'] = tbs
    if safe:
        querystring['safe'] = safe
    if exclude_if_modified:
        querystring['exclude_if_modified'] = exclude_if_modified
    if show_duplicates:
        querystring['show_duplicates'] = show_duplicates
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_search(q: str, engine: str, num: int=None, safe: str=None, page: int=None, hl: str=None, time_period: str=None, cr: str=None, knowledge_graph_id: str=None, time_period_max: str=None, nfpr: int=None, tbs: str=None, filter: str=None, location: str=None, time_period_min: str=None, flatten_results: bool=None, include_answer_box: str=None, gl: str=None, google_domain: str=None, location_auto: bool=None, uule: str=None, lr: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Search Parameters are applicable when making a request with `engine=google` to retrieve Google search results for a given search term. The search term is specified in the `q` parameter. The location your search is run from is determined by the location parameter, which can be populated with a `full_name` value from the Locations API."
    q: The keyword you want to use to perform the search.
        engine: Should be set to `engine=google`.

        num: Determines the number of results to show per page. Use in combination with the `page` parameter to implement pagination

        safe: Determines whether Safe Search is enabled for the results. Can be set to `active` to enable Safe Search, or `off` to disable Safe Search.

        page: Determines the page of results to return, defaults to `1`. Use in combination with the `num` parameter to implement pagination.

        hl: The `hl` parameter determines the Google UI language to return results. View the full list of supported `hl` values [here](https://www.serpwow.com/docs/search-api/reference/google-languages). Defaults to `en`.

        time_period: Determines the time period of the results shown. It can be set to `last_hour`, `last_day` (for the last 24 hours), `last_week` (for the last 7 days), `last_month`, `last_year` or `custom`. When using `custom` you must also specifiy one or both of the `time_period_minor` `time_period_max` parameters to define the custom time period.

        cr: The `cr` parameter instructs Google to limit the results to websites in the specified country. View the full list of supported `cr` values [here](https://www.serpwow.com/docs/search-api/reference/google-cr-countries).

        knowledge_graph_id: The `knowledge_graph_id` request parameter sets the `kgmid` Google parameter. You can use this to prompt a specific knowledge graph to show in the results, an example would be `knowledge_graph_id=/m/0jg24`

        time_period_max: Determines the maximum (i.e. 'to') time to use when `time_period` is set to custom. Should be in the form `MM/DD/YYYY,` I.e. for 31st December 2018 `time_period_max` would be `12/31/2018`.

        nfpr: Determines whether to exclude results from auto-corrected queries that were spelt wrong. Can be set to `1` to exclude auto-corrected results, or `0` (default) to include them.

        tbs: Sets a specific string to be added to the Google `tbs` parameter in the underlying Google query. The `tbs` parameter is normally generated automatically by the API, but it can be set explicitly also.

        filter: Determines if the filters for `Similar Results` and `Omitted Results` are on or off. Can be set to `1` (default) to enable these filters, or `0` to disable these filters.

        location: Determines the geographic location in which the query is executed. You can enter any location as free-text, but if you choose one of the SerpWow [built-in locations](https://www.serpwow.com/docs/locations-api) then the `google_domain`, `gl` and `hl` parameters are automatically updated to the domain, country and language that match the built-in location (note that this behaviour can be disabled via the `location_auto `parameter).

        time_period_min: Determines the minimum (i.e. 'from') time to use when time_periodis set to `custom`. Should be in the form `MM/DD/YYYY`, I.e. for 31st December 2018 `time_period_min` would be `12/31/2018`.

        flatten_results: Can be set to `true` or `false`. Determines whether SerpWow flattens the `inline_videos`, `inline_images`, `inline_tweets`, `top_stories` and `local_results` and shows them inline with the `organic_results`. This is useful if you want a simplified list of all of the results shown for an organic web search, irrespective of the type of result. When `flatten_results=true` then a new property type is added to each item in the organic_results array indicating the type of result (i.e. \\\"ad\\\", \\\"inline_tweets\\\" etc).

        include_answer_box: Determines whether to include the answer box (sometimes called the \\\"featured snippet\\\") in the `organic_results` array and treat it as the first result. This may be desirable if you treat the result Bing displayed in the `answer_box` as the first organic result.

        gl: The gl parameter determines the Google country to use for the query. View the full list of supported `gl` values [here](The gl parameter determines the Google country to use for the query. View the full list of supported glvalues here. Defaults to us.
). Defaults to `us`.

        google_domain: The Google domain to use to run the search query. View the full list of supported `google_domain` values [here](https://www.serpwow.com/docs/search-api/reference/google-domains). Defaults to `google.com`.

        location_auto: If the `location` field is set to a SerpWow [built-in location](https://www.serpwow.com/docs/locations-api) from the [Locations](https://www.serpwow.com/docs/locations-api) API, and `location_auto` is set to true (default) then the google_domain, gland hlparameters are automatically updated to the domain, country and language that match the built-in `location`. Valid values are `true` (default) to enable this behaviour or `false` to disable.

        uule: The Google UULE parameter - use to pass through a custom `uule` parameter to Google. SerpWow automatically generates the `uule` when you use the `location` parameter but we allow you to overwrite it directly by specifying a `uule` directly.

        lr: The `lr` parameter limits the results to websites containing the specified language. View the full list of supported `lr` values [here](https://www.serpwow.com/docs/search-api/reference/google-lr-languages).

        
    """
    url = f"https://serpwow.p.rapidapi.com/search"
    querystring = {'q': q, 'engine': engine, }
    if num:
        querystring['num'] = num
    if safe:
        querystring['safe'] = safe
    if page:
        querystring['page'] = page
    if hl:
        querystring['hl'] = hl
    if time_period:
        querystring['time_period'] = time_period
    if cr:
        querystring['cr'] = cr
    if knowledge_graph_id:
        querystring['knowledge_graph_id'] = knowledge_graph_id
    if time_period_max:
        querystring['time_period_max'] = time_period_max
    if nfpr:
        querystring['nfpr'] = nfpr
    if tbs:
        querystring['tbs'] = tbs
    if filter:
        querystring['filter'] = filter
    if location:
        querystring['location'] = location
    if time_period_min:
        querystring['time_period_min'] = time_period_min
    if flatten_results:
        querystring['flatten_results'] = flatten_results
    if include_answer_box:
        querystring['include_answer_box'] = include_answer_box
    if gl:
        querystring['gl'] = gl
    if google_domain:
        querystring['google_domain'] = google_domain
    if location_auto:
        querystring['location_auto'] = location_auto
    if uule:
        querystring['uule'] = uule
    if lr:
        querystring['lr'] = lr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serpwow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

