import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_realtime_customizations_of_a_menu_item(product_id: str, image_height: int=300, pickup: bool=None, user_latitude: int=None, user_street_num: str='188', image_width: int=300, user_zipcode: str='94107', user_longitude: int=None, include_default_customization_description: bool=None, user_country: str='US', user_city: str='San Francisco', user_state: str='CA', user_street_name: str='King Street', include_quote: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    product_id: The Product ID given for this store by the Restaurant Menu API or Grocery Products API
        image_height: The desired height to scale the product's image to. If supplied, then `image_width` will be ignored. The `thumbnail_image` property in the response will represent the scaled image URL, if `supports_image_scaling=true`.
        pickup: `true` for pickup quotes, `false` for delivery quotes.
        user_latitude: The latitude of the user. Strongly recommended for realtime price and availability.
        user_street_num: Street number of the user.
        image_width: The desired width to scale product's image to. If `image_height` is supplied, then `image_width` will be ignored. The `thumbnail_image` property in the response will represent the scaled image URL, if `supports_image_scaling=true`.
        user_zipcode: Zipcode of the user.
        user_longitude: The longitude of the user. Strongly recommended for realtime price and availability. In the US, this value is negative.
        include_default_customization_description: Whether to include a description of included customizations.
        user_country: Country of the user. Can be `US` or `CA`.
        user_city: City of the user.
        user_state: State abbreviation of the user.
        user_street_name: Street name of the user.
        include_quote: Whether to include the realtime delivery/pickup quote.
        
    """
    url = f"https://menumy-menu-api3.p.rapidapi.com/details/product"
    querystring = {'product_id': product_id, }
    if image_height:
        querystring['image_height'] = image_height
    if pickup:
        querystring['pickup'] = pickup
    if user_latitude:
        querystring['user_latitude'] = user_latitude
    if user_street_num:
        querystring['user_street_num'] = user_street_num
    if image_width:
        querystring['image_width'] = image_width
    if user_zipcode:
        querystring['user_zipcode'] = user_zipcode
    if user_longitude:
        querystring['user_longitude'] = user_longitude
    if include_default_customization_description:
        querystring['include_default_customization_description'] = include_default_customization_description
    if user_country:
        querystring['user_country'] = user_country
    if user_city:
        querystring['user_city'] = user_city
    if user_state:
        querystring['user_state'] = user_state
    if user_street_name:
        querystring['user_street_name'] = user_street_name
    if include_quote:
        querystring['include_quote'] = include_quote
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menumy-menu-api3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_realtime_menu_of_a_restaurant_inventory_of_a_grocery_store(user_zipcode: str, user_state: str, user_longitude: int, user_country: str, user_street_name: str, user_latitude: int, store_id: str, user_city: str, user_street_num: str, image_height: int=300, image_width: int=300, menu_id: str=None, quote_preference: str='first_available', budget: int=20, include_quote: bool=None, include_customizations: bool=None, pickup: bool=None, subcategory_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    user_zipcode: Zipcode of the user.
        user_state: State abbreviation of the user.
        user_longitude: The longitude of the user. In the US, this value is negative.
        user_country: Country of the user. Can be `US` or `CA`.
        user_street_name: Street name of the user.
        user_latitude: The latitude of the user.
        store_id: The ID if the store to fetch a menu for
        user_city: City of the user.
        user_street_num: Street number of the user.
        image_height: The desired height to scale product images to. If supplied, then `image_width` will be ignored. The `thumbnail_image` property in the response will represent the scaled image URL, if `supports_image_scaling=true`.
        image_width: The desired width to scale product images to. If `image_height` is supplied, then `image_width` will be ignored. The `thumbnail_image` property in the response will represent the scaled image URL, if `supports_image_scaling=true`.
        menu_id: The ID of the menu to fetch. If supplied, the `store_id` and `quote_preference` parameters will be ignored.
        quote_preference: The preferred quote for order fulfillment. Can be `default`, `cheapest`, `fastest`, `first_available`, `cheapest_inventory`, or `cheapest_for_store`.
        budget: The maximum amount, in Dollars, that the user has specified they'd like to spend on an order. Used when `quote_preference=cheapest`.
        include_quote: Whether to include the realtime delivery/pickup quote.
        include_customizations: Whether to include customizations.
        pickup: `true` for pickup quotes, `false` for delivery quotes.
        subcategory_id: The optional Subcategory ID, needed for some requests if the menu is too large.
        
    """
    url = f"https://menumy-menu-api3.p.rapidapi.com/details/inventory"
    querystring = {'user_zipcode': user_zipcode, 'user_state': user_state, 'user_longitude': user_longitude, 'user_country': user_country, 'user_street_name': user_street_name, 'user_latitude': user_latitude, 'store_id': store_id, 'user_city': user_city, 'user_street_num': user_street_num, }
    if image_height:
        querystring['image_height'] = image_height
    if image_width:
        querystring['image_width'] = image_width
    if menu_id:
        querystring['menu_id'] = menu_id
    if quote_preference:
        querystring['quote_preference'] = quote_preference
    if budget:
        querystring['budget'] = budget
    if include_quote:
        querystring['include_quote'] = include_quote
    if include_customizations:
        querystring['include_customizations'] = include_customizations
    if pickup:
        querystring['pickup'] = pickup
    if subcategory_id:
        querystring['subcategory_id'] = subcategory_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menumy-menu-api3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_a_restaurant_or_grocery_store(latitude: int, longitude: int, include_utc_hours: bool=None, search_focus: str='store', min_lon: int=None, max_lon: int=None, min_lat: int=None, max_lat: int=None, maximum_miles: int=3, default_quote: bool=None, dollar_signs: int=None, page: int=None, cuisine: str=None, autocomplete: bool=None, open: bool=None, fetch_quotes: bool=None, user_state: str='CA', timeout: int=None, pickup: bool=None, user_street_name: str='King Street', user_zipcode: str='94107', sort: str='relevance', user_country: str='US', user_street_num: str='188', minimum_rating: int=None, store_type: str='restaurant', user_city: str='San Francisco', budget: int=20, query: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    latitude: The latitude of the user.
        longitude: The longitude of the user. In the US, this value is negative.
        include_utc_hours: Whether to include store hours in utc time.
        search_focus: Whether to focus the search on store name or item name. Can be `item`, `store`, or left empty for both.
        min_lon: The minimum longitude for a "Search In This Area".
        max_lon: The maximum longitude for a "Search In This Area".
        min_lat: The minimum latitude for a "Search In This Area".
        max_lat: The maximum latitude for a "Search In This Area".
        maximum_miles: ex: `1.5`. The maximum distance allowed, in miles.
        default_quote: Whether to return one default `quote` when `fetch_quotes` is `true`.
        dollar_signs: ex: `2` (which would signal $$). Cost for the average meal. This can represent `1` ($) for the cheapest stores or `4` ($$$$) for the most expensive stores.
        page: ex: `0`. The page to fetch, to retrieve more results. This is returned as `next_page` in the response.
        cuisine: The cuisine to filter stores by.
        autocomplete: Whether to autocomplete the query.
        open: Whether or not to return only open restaurants.
        fetch_quotes: Whether to fetch realtime delivery/pickup quotes. Note that this being `true` greatly increases response time. Note that if this is `true` and the `query` is also empty then store hours will not be in the response.
        user_state: State abbreviation of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        timeout: Timeout to apply if `fetch_quotes = true`, in seconds.
        pickup: If `true`, filter stores to only those that offer pickup. If `false`, filter to only stores with delivery.
        user_street_name: Street name of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        user_zipcode: Zipcode of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        sort: The sort type. Can be `cheapest`, `fastest`, `rating`, `distance` or the default `relevance`.
        user_country: Country of the user. Can be `US` or `CA`. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        user_street_num: Street number of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        minimum_rating: ex: `4.6`. The minimum rating allowed.
        store_type: The type of stores to search from. Can be `restaurant`, `grocery`, or empty string for both.
        user_city: City of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        budget: The maximum amount, in Dollars, that the user has specified they'd like to spend on this meal. Required for applying a `sort`.
        query: The search query. Try leaving this empty to browse what's available!
        
    """
    url = f"https://menumy-menu-api3.p.rapidapi.com/search/store/v3"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if include_utc_hours:
        querystring['include_utc_hours'] = include_utc_hours
    if search_focus:
        querystring['search_focus'] = search_focus
    if min_lon:
        querystring['min_lon'] = min_lon
    if max_lon:
        querystring['max_lon'] = max_lon
    if min_lat:
        querystring['min_lat'] = min_lat
    if max_lat:
        querystring['max_lat'] = max_lat
    if maximum_miles:
        querystring['maximum_miles'] = maximum_miles
    if default_quote:
        querystring['default_quote'] = default_quote
    if dollar_signs:
        querystring['dollar_signs'] = dollar_signs
    if page:
        querystring['page'] = page
    if cuisine:
        querystring['cuisine'] = cuisine
    if autocomplete:
        querystring['autocomplete'] = autocomplete
    if open:
        querystring['open'] = open
    if fetch_quotes:
        querystring['fetch_quotes'] = fetch_quotes
    if user_state:
        querystring['user_state'] = user_state
    if timeout:
        querystring['timeout'] = timeout
    if pickup:
        querystring['pickup'] = pickup
    if user_street_name:
        querystring['user_street_name'] = user_street_name
    if user_zipcode:
        querystring['user_zipcode'] = user_zipcode
    if sort:
        querystring['sort'] = sort
    if user_country:
        querystring['user_country'] = user_country
    if user_street_num:
        querystring['user_street_num'] = user_street_num
    if minimum_rating:
        querystring['minimum_rating'] = minimum_rating
    if store_type:
        querystring['store_type'] = store_type
    if user_city:
        querystring['user_city'] = user_city
    if budget:
        querystring['budget'] = budget
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menumy-menu-api3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_a_menu_item_or_grocery_product(user_latitude: int, user_longitude: int, max_lat: int=None, min_lon: int=None, search_in_category: str=None, autocomplete: bool=None, minimum_rating: int=None, page: int=None, maximum_miles: int=None, open: bool=None, dollar_signs: int=None, budget: int=20, query: str='asparagus', product_id: str=None, max_lon: int=None, min_lat: int=None, menu_id: str=None, pickup: bool=None, cuisine: str=None, sort: str='relevance', fuzzy_search: bool=None, sale: bool=None, user_street_num: str='188', user_city: str='San Francisco', user_country: str='US', upc_codes: str=None, store_name: str=None, user_street_name: str='King Street', user_zipcode: str='94107', fetch_quotes: bool=None, user_state: str='CA', store_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    user_latitude: The latitude of the user.
        user_longitude: The longitude of the user. In the US, this value is negative.
        max_lat: The maximum latitude for a "Search In This Area".
        min_lon: The minimum longitude for a "Search In This Area".
        search_in_category: The category of the menu to search for products within. The list of a menu's categories are provided by `/details/inventory`.
        autocomplete: Whether to autocomplete the query.
        minimum_rating: ex: `4.6`. The minimum rating allowed.
        page: ex: 0. The page to fetch
        maximum_miles: ex: `1.5`. The maximum distance allowed, in miles.
        open: Whether to return only open restaurants.
        dollar_signs: ex: `2` (which would signal $$). Cost for the average meal. This can represent `1` ($) for the cheapest stores or `4` ($$$$) for the most expensive stores.
        budget: The maximum amount, in Dollars, that the user has specified they'd like to spend on an order. Required for applying a `cheapest` sort.
        query: The product search query. If empty, this endpoint will let you browse products available for ordering nearby.
        product_id: Optional. If supplied, `query` will be ignored, and this endpoint will search for all matching products close to the user's location within the `maximum_miles` radius. A matching product is one that perfectly matches the same brand (i.e. Safeway) and product (i.e. Almond Breeze Unsweetened Original Almondmilk). Note: `product_id` can retrieved from `/details/inventory`, `/search/product`, and `/search/cart`
        max_lon: The maximum longitude for a "Search In This Area".
        min_lat: The minimum latitude for a "Search In This Area".
        menu_id: The ID of the menu from which to search for products within. The menu ID is provided by `/details/inventory`.
        pickup: If `true`, filter stores to only those that offer pickup. If `false`, filter to only stores with delivery.
        cuisine: The cuisine to filter products by.
        sort: The sort type. Can be `cheapest`, `fastest`, `rating`, `distance` or the default `relevance`.
        fuzzy_search: Whether to perform a lenient search that returns relevant products that may not exactly match the `query` provided.
        sale: Whether to search for products that are currently on sale.
        user_street_num: Street number of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        user_city: City of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        user_country: Country of the user. Can be `US` or `CA`. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        upc_codes: Example: `66538783,031146250103`. The UPC codes to search for, delimited by comma. Takes precedence over the other ways you can search, `query` and `product_id`.
        store_name: The store from which to search for products within.
        user_street_name: Street name of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        user_zipcode: Zipcode of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        fetch_quotes: Whether to fetch realtime delivery/pickup quotes. Note that this being `true` greatly increases response time. Note that if this is `true` and the `query` is also empty then store hours will not be in the response.
        user_state: State abbreviation of the user. Required if `fetch_quotes` is `true` and `pickup` is `false`.
        store_type: The type of stores to search from. Can be `grocery`, `restaurant`, or empty string for both.
        
    """
    url = f"https://menumy-menu-api3.p.rapidapi.com/search/product/v4"
    querystring = {'user_latitude': user_latitude, 'user_longitude': user_longitude, }
    if max_lat:
        querystring['max_lat'] = max_lat
    if min_lon:
        querystring['min_lon'] = min_lon
    if search_in_category:
        querystring['search_in_category'] = search_in_category
    if autocomplete:
        querystring['autocomplete'] = autocomplete
    if minimum_rating:
        querystring['minimum_rating'] = minimum_rating
    if page:
        querystring['page'] = page
    if maximum_miles:
        querystring['maximum_miles'] = maximum_miles
    if open:
        querystring['open'] = open
    if dollar_signs:
        querystring['dollar_signs'] = dollar_signs
    if budget:
        querystring['budget'] = budget
    if query:
        querystring['query'] = query
    if product_id:
        querystring['product_id'] = product_id
    if max_lon:
        querystring['max_lon'] = max_lon
    if min_lat:
        querystring['min_lat'] = min_lat
    if menu_id:
        querystring['menu_id'] = menu_id
    if pickup:
        querystring['pickup'] = pickup
    if cuisine:
        querystring['cuisine'] = cuisine
    if sort:
        querystring['sort'] = sort
    if fuzzy_search:
        querystring['fuzzy_search'] = fuzzy_search
    if sale:
        querystring['sale'] = sale
    if user_street_num:
        querystring['user_street_num'] = user_street_num
    if user_city:
        querystring['user_city'] = user_city
    if user_country:
        querystring['user_country'] = user_country
    if upc_codes:
        querystring['upc_codes'] = upc_codes
    if store_name:
        querystring['store_name'] = store_name
    if user_street_name:
        querystring['user_street_name'] = user_street_name
    if user_zipcode:
        querystring['user_zipcode'] = user_zipcode
    if fetch_quotes:
        querystring['fetch_quotes'] = fetch_quotes
    if user_state:
        querystring['user_state'] = user_state
    if store_type:
        querystring['store_type'] = store_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "menumy-menu-api3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

