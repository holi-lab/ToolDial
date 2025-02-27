import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_ultra(services: str, country: str, max_imdb_vote_count: int=1000000, order_by: str='year', max_imdb_rating: int=90, min_imdb_rating: int=70, desc: bool=None, keyword: str=None, cursor: str=None, year_max: int=2025, min_imdb_vote_count: int=10000, year_min: int=2000, show_original_language: str='en', genres_relation: str='or', genres: str='18,80', output_language: str='en', show_type: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through the catalog of the given services in the given country. Provides filters such as show language, genre and keyword. Items per page is 25. Output includes all the information about the shows, such as title, IMDb ID, TMDb ID, IMDb rating, link to shows' pages in streaming services (including individual season/episode links), cast, runtime, poster and many more! Apart from the info about the given country-service combinations, output also includes information about availability in the other services for the given country."
    services: A comma separated list of up to 16 services to search in. See `/v2/services` endpoint to get the supported services and their ids/names.

Syntax of the values supplied in the list can be as the followings:

`<sevice_id>`: Searches in the entire catalog of that service, including (if applicable) rentable, buyable shows or shows available through addons i.e. `netflix`, `prime`, `apple`

`<sevice_id>.<offer_type>`: Only returns the shows that are available in that service with the given offer type. Valid offer type values are `subscription`, `free`, `rent`, `buy` and `addon`  i.e. `peacock.free` only returns the shows on Peacock that are free to watch, `prime.subscription` only returns the shows on Prime Video that are available to watch with a Prime subscription. `hulu.addon` only returns the shows on Hulu that are available via an addon,  `prime.rent` only returns the shows on Prime Video that are rentable.

`<sevice_id>.addon.<addon_id>`: Only returns the shows that are available in that service with the given addon. Check `/v2/services` endpoint to fetch the available addons for a service. Some sample values are: `hulu.addon.hbo`,  `prime.addon.hbomaxus`.

If a service supports both `free` and `subscription`, then results included under `subscription` will always include the `free` shows as well.

When multiple values are passed as a comma separated list, any show that satisfies at least one of the values will be included in the result.

Some sample list values:

`prime.rent,prime.buy,apple.rent,apple.buy`: Returns all the buyable/rentable shows on Amazon Prime Video and Apple TV.

`prime.addon,prime.subscription`: Returns all the shows on Amazon Prime Video that are available through either a Prime Video subscription or a Prime Video Channel subscription.
        country: 2 letter ISO 3166-1 alpha-2 country code of the country to search in. See the about page to check the supported countries.
        order_by: Determines the ordering of the results.

Possible values are `original_title`, `imdb_vote_count`, `imdb_rating` and `year`. Default value is `original_title`
        max_imdb_rating: Out of 100
        min_imdb_rating: Out of 100
        desc: Use descending order?

Possible values are `true` and `false`. Default value is `false`.
        keyword: A keyword to only search within the shows have that keyword in their overview or title.
        cursor: Cursor is used for pagination. After each request, the response includes a `hasMore` boolean field to tell if there are more results that did not fit the returned list size. If it is set as true, to get the rest of the result set, send a new request (with the same parameters for other fields such as show_type, services etc.), and set the cursor parameter as the `nextCursor` value of the previous request response. Do not forget to escape the cursor value before putting it into the query as it might contain characters such as `?`, `&`.

The first request naturally does not require a `cursor` parameter.
        year_max: Maximum release/air year of the show.
        year_min: Minimum release/air year of the show.
        show_original_language: A 2 letter ISO 639-1 language code to only search within the shows whose original language matches with the provided language.
        genres_relation: When more than one genre is supplied in `genres` parameter, `genres_relation` chooses the matching style against

When `or`, the endpoint returns any show that has at least one of the given genres.
When `and`, it only returns the shows that have all of the given genres.

Default value is `and`.
        genres: Comma separated list of genre ids to only search within the shows in those genres. See `/v2/genres` endpoint to see available genres and ids.

When more than one genre supplied, set `genres_relation` parameter to specify between returning shows that have at least one of the given genres or returning shows that have all of the given genres
        output_language: 2 letter iso code of the output language. Default is `en`. See the about page to see the list of languages supported.
        show_type: Type of shows to search in. Accepted values are `movie`, `series` or `all`. The default value is `all`.
        
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/search/ultra"
    querystring = {'services': services, 'country': country, }
    if max_imdb_vote_count:
        querystring['max_imdb_vote_count'] = max_imdb_vote_count
    if order_by:
        querystring['order_by'] = order_by
    if max_imdb_rating:
        querystring['max_imdb_rating'] = max_imdb_rating
    if min_imdb_rating:
        querystring['min_imdb_rating'] = min_imdb_rating
    if desc:
        querystring['desc'] = desc
    if keyword:
        querystring['keyword'] = keyword
    if cursor:
        querystring['cursor'] = cursor
    if year_max:
        querystring['year_max'] = year_max
    if min_imdb_vote_count:
        querystring['min_imdb_vote_count'] = min_imdb_vote_count
    if year_min:
        querystring['year_min'] = year_min
    if show_original_language:
        querystring['show_original_language'] = show_original_language
    if genres_relation:
        querystring['genres_relation'] = genres_relation
    if genres:
        querystring['genres'] = genres
    if output_language:
        querystring['output_language'] = output_language
    if show_type:
        querystring['show_type'] = show_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_basic_free(services: str, country: str, cursor: str=None, show_original_language: str='en', genre: str='18', keyword: str='zombie', output_language: str='en', show_type: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through the catalog of the given services in the given country. Provides filters such as show language, genre and keyword. Items per page is 8. Output includes all the information about the shows, such as title, IMDb ID, TMDb ID, IMDb rating, link to shows' pages in streaming services (including individual season/episode links), cast, runtime, poster and many more! Apart from the info about the given country-service combinations, output also includes information about availability in the other services for the given country."
    services: A comma separated list of up to 4 services to search in. See `/v2/services` endpoint to get the supported services and their ids/names.

Syntax of the values supplied in the list can be as the followings:

`<sevice_id>`: Searches in the entire catalog of that service, including (if applicable) rentable, buyable shows or shows available through addons i.e. `netflix`, `prime`, `apple`

`<sevice_id>.<offer_type>`: Only returns the shows that are available in that service with the given offer type. Valid offer type values are `subscription`, `free`, `rent`, `buy` and `addon`  i.e. `peacock.free` only returns the shows on Peacock that are free to watch, `prime.subscription` only returns the shows on Prime Video that are available to watch with a Prime subscription. `hulu.addon` only returns the shows on Hulu that are available via an addon,  `prime.rent` only returns the shows on Prime Video that are rentable.

`<sevice_id>.addon.<addon_id>`: Only returns the shows that are available in that service with the given addon. Check `/v2/services` endpoint to fetch the available addons for a service. Some sample values are: `hulu.addon.hbo`,  `prime.addon.hbomaxus`.

If a service supports both `free` and `subscription`, then results included under `subscription` will always include the `free` shows as well.

When multiple values are passed as a comma separated list, any show that satisfies at least one of the values will be included in the result.

Some sample list values:

`prime.rent,prime.buy,apple.rent,apple.buy`: Returns all the buyable/rentable shows on Amazon Prime Video and Apple TV.

`prime.addon,prime.subscription`: Returns all the shows on Amazon Prime Video that are available through either a Prime Video subscription or a Prime Video Channel subscription.
        country: 2 letter ISO 3166-1 alpha-2 country code of the country to search in. See the about page to check the supported countries.
        cursor: Cursor is used for pagination. After each request, the response includes a `hasMore` boolean field to tell if there are more results that did not fit the returned list size. If it is set as true, to get the rest of the result set, send a new request (with the same parameters for other fields such as show_type, services etc.), and set the cursor parameter as the `nextCursor` value of the previous request response. Do not forget to escape the cursor value before putting it into the query as it might contain characters such as `?`, `&`.

The first request naturally does not require a `cursor` parameter.
        show_original_language: A 2 letter ISO 639-1 language code to only search within the shows whose original language matches with the provided language.
        genre: A genre id to only search within the shows in that genre. See `/v2/genres` endpoint to see available genres and ids.
        keyword: A keyword to only search within the shows have that keyword in their overview or title.
        output_language: 2 letter iso code of the output language. Default is `en`. See the about page to see the list of languages supported.
        show_type: Type of shows to search in. Accepted values are `movie`, `series` or `all`. The default value is `all`.
        
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/search/basic"
    querystring = {'services': services, 'country': country, }
    if cursor:
        querystring['cursor'] = cursor
    if show_original_language:
        querystring['show_original_language'] = show_original_language
    if genre:
        querystring['genre'] = genre
    if keyword:
        querystring['keyword'] = keyword
    if output_language:
        querystring['output_language'] = output_language
    if show_type:
        querystring['show_type'] = show_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def services_free(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of supported services and details about them"
    
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/services"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_basic_free(country: str, imdb_id: str='tt1877830', output_language: str='en', tmdb_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a show via IMDb or TMDb id, including the streaming availability info in the given country."
    output_language: 2 letter iso code of the output language. Default is `en`. See the about page to see the list of languages supported.
        
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/get/basic"
    querystring = {'country': country, }
    if imdb_id:
        querystring['imdb_id'] = imdb_id
    if output_language:
        querystring['output_language'] = output_language
    if tmdb_id:
        querystring['tmdb_id'] = tmdb_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_pro(country: str, services: str, year_max: int=2025, cursor: str=None, desc: bool=None, genres_relation: str='or', year_min: int=2000, show_original_language: str='en', genres: str='18,80', show_type: str='movie', order_by: str='year', output_language: str='en', keyword: str='zombie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through the catalog of the given services in the given country. Provides filters such as show language, genre and keyword. Items per page is 10. Output includes all the information about the shows, such as title, IMDb ID, TMDb ID, IMDb rating, link to shows' pages in streaming services (including individual season/episode links), cast, runtime, poster and many more! Apart from the info about the given country-service combinations, output also includes information about availability in the other services for the given country."
    country: 2 letter ISO 3166-1 alpha-2 country code of the country to search in. See the about page to check the supported countries.
        services: A comma separated list of up to 8 services to search in. See `/v2/services` endpoint to get the supported services and their ids/names.

Syntax of the values supplied in the list can be as the followings:

`<sevice_id>`: Searches in the entire catalog of that service, including (if applicable) rentable, buyable shows or shows available through addons i.e. `netflix`, `prime`, `apple`

`<sevice_id>.<offer_type>`: Only returns the shows that are available in that service with the given offer type. Valid offer type values are `subscription`, `free`, `rent`, `buy` and `addon`  i.e. `peacock.free` only returns the shows on Peacock that are free to watch, `prime.subscription` only returns the shows on Prime Video that are available to watch with a Prime subscription. `hulu.addon` only returns the shows on Hulu that are available via an addon,  `prime.rent` only returns the shows on Prime Video that are rentable.

`<sevice_id>.addon.<addon_id>`: Only returns the shows that are available in that service with the given addon. Check `/v2/services` endpoint to fetch the available addons for a service. Some sample values are: `hulu.addon.hbo`,  `prime.addon.hbomaxus`.

If a service supports both `free` and `subscription`, then results included under `subscription` will always include the `free` shows as well.

When multiple values are passed as a comma separated list, any show that satisfies at least one of the values will be included in the result.

Some sample list values:

`prime.rent,prime.buy,apple.rent,apple.buy`: Returns all the buyable/rentable shows on Amazon Prime Video and Apple TV.

`prime.addon,prime.subscription`: Returns all the shows on Amazon Prime Video that are available through either a Prime Video subscription or a Prime Video Channel subscription.
        year_max: Maximum release/air year of the show.
        cursor: Cursor is used for pagination. After each request, the response includes a `hasMore` boolean field to tell if there are more results that did not fit the returned list size. If it is set as true, to get the rest of the result set, send a new request (with the same parameters for other fields such as show_type, services etc.), and set the cursor parameter as the `nextCursor` value of the previous request response. Do not forget to escape the cursor value before putting it into the query as it might contain characters such as `?`, `&`.

The first request naturally does not require a `cursor` parameter.
        desc: Use descending order?

Possible values are `true` and `false`. Default value is `false`.
        genres_relation: When more than one genre is supplied in `genres` parameter, `genres_relation` chooses the matching style against

When `or`, the endpoint returns any show that has at least one of the given genres.
When `and`, it only returns the shows that have all of the given genres.

Default value is `and`.
        year_min: Minimum release/air year of the show.
        show_original_language: A 2 letter ISO 639-1 language code to only search within the shows whose original language matches with the provided language.
        genres: Comma separated list of genre ids to only search within the shows in those genres. See `/v2/genres` endpoint to see available genres and ids.

When more than one genre supplied, set `genres_relation` parameter to specify between returning shows that have at least one of the given genres or returning shows that have all of the given genres
        show_type: Type of shows to search in. Accepted values are `movie`, `series` or `all`. The default value is `all`.
        order_by: Determines the ordering of the results.

Possible values are `original_title` and `year`. Default value is `original_title`
        output_language: 2 letter iso code of the output language. Default is `en`. See the about page to see the list of languages supported.
        keyword: A keyword to only search within the shows have that keyword in their overview or title.
        
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/search/pro"
    querystring = {'country': country, 'services': services, }
    if year_max:
        querystring['year_max'] = year_max
    if cursor:
        querystring['cursor'] = cursor
    if desc:
        querystring['desc'] = desc
    if genres_relation:
        querystring['genres_relation'] = genres_relation
    if year_min:
        querystring['year_min'] = year_min
    if show_original_language:
        querystring['show_original_language'] = show_original_language
    if genres:
        querystring['genres'] = genres
    if show_type:
        querystring['show_type'] = show_type
    if order_by:
        querystring['order_by'] = order_by
    if output_language:
        querystring['output_language'] = output_language
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ultra(output_language: str='en', tmdb_id: str=None, imdb_id: str='tt1877830', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a show via IMDb or TMDb id, including the global streaming availability info"
    output_language: 2 letter iso code of the output language. Default is `en`. See the about page to see the list of languages supported.
        
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/get/ultra"
    querystring = {}
    if output_language:
        querystring['output_language'] = output_language
    if tmdb_id:
        querystring['tmdb_id'] = tmdb_id
    if imdb_id:
        querystring['imdb_id'] = imdb_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_title_free(title: str, country: str, show_type: str='movie', output_language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search movies and series by title. Maximum amount of items returned are 20, unless there are more than 20 shows with the exact given title input. In that case all the items have 100% match with the title will be returned.
		
		No pagination is supported."
    title: Title to search for.
        country: Regardless of this value, the given title is searched across all the platforms and all the countries. This parameter determines according to which country the streamingInfo field will be populated. Thus even if a show is not available in this country, It will be included in the results if it resembles with the given title, but the streaming information will not be available.
        show_type: Type of shows to include in the results. Either `movie`, `series` or `all`. Default is `all`.
        output_language: 2 letter iso code of the output language. Default is `en`. See the about page to see the list of languages supported.
        
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/search/title"
    querystring = {'title': title, 'country': country, }
    if show_type:
        querystring['show_type'] = show_type
    if output_language:
        querystring['output_language'] = output_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genres_free(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the id to name mapping of supported genres."
    
    """
    url = f"https://streaming-availability.p.rapidapi.com/v2/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

