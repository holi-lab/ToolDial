import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def category(category_id: str, type: str, delivery_type: str=None, url: str=None, facets: str=None, condition: str=None, rating: str=None, min_price: str=None, sort_by: str=None, page: str=None, max_price: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Category Parameters are applicable when making a request with type=category to retrieve category results from Walmart. The category is specified in the category_id parameter. The parameters should be appended as querystring parameters to the Walmart Product Data APIGETHTTP request.
		
		Category results are retrieved from the Category results page on Walmart."
    category_id: 
A category ID to retrieve results from. You may supply any arbitary value in the category_id parameter however we recommend using a category ID returned from the Categories API as these are known-good category IDs from Walmart.

In the case of the category_id parameter not matching a value from the Categories API, BlueCart will append the passed category_id value in the cat_id Walmart URL querystring parameter.
        delivery_type: 
Determines whether results are filtered to just those matching the given delivery type. Valid values are:
pickup
Filter results to products available for pickup in store.
delivery_from_store
Filter results to products available delivery from a store.
shipping
Filter results to products available via home shipping.
available_in_store
Filter results to products available in store (use the customer_zipcode parameter to set the store location).

        url: The Walmart category results page to retrieve results from.

Note: If the url parameter is supplied then all other parameters are ignored as they are overriden be those specified in the url parameter.
        facets: A free-form set of filter facets to use with the request. Available facet data is returned from `type=category` requests or can be retrieved from any Walmart URL in the facet Walmart URL querystring parameter.

Facet data should be expressed in comma seperated notation, with the name and value of the facet delimited by an equals `(\\\\\\\"=\\\\\\\")` character.

For example, to set the brand facet to the value `Garmin` and the `screen_size` facet to the value `6\\\\\\\"` the value of the `facets` request parameter would be `facets=brand=Garmin,screen_size=6\\\\\\\"`
        condition: 
Filters the results to just products matching the given condition. Valid values are:
new
Filter results to just new products.
refurbished
Filter results to just refurbished products.
        rating: Determines whether results are filtered to just those matching the given customer star rating. Valid values are:
one_star
Filter results to products with a 1 to 1.9 star customer rating.
two_star
Filter results to products with a 2 to 2.9 star customer rating.
three_star
Filter results to products with a 3 to 3.9 star customer rating.
four_star
Filter results to products with a 4 to 5 star customer rating.
        min_price: Determines the minimum price of results to return, expressed in dollars and cents (i.e. `min_price=2.99` for $2.99).
        sort_by: Determines how the results are sorted. Valid values are:

- best_seller
- price_high_to_low
- price_low_to_high
- best_match
- highest_rating
- newly_listed
- new
        page: The current page of results to retrieve. Inspect the pagination results property for details on the number of pages available.
update value of num to get additional pages
        max_price: Determines the maximum price of results to return, expressed in dollars and cents (i.e. `min_price=39.99` for $39.99).
        
    """
    url = f"https://bluecart.p.rapidapi.com/request"
    querystring = {'category_id': category_id, 'type': type, }
    if delivery_type:
        querystring['delivery_type'] = delivery_type
    if url:
        querystring['url'] = url
    if facets:
        querystring['facets'] = facets
    if condition:
        querystring['condition'] = condition
    if rating:
        querystring['rating'] = rating
    if min_price:
        querystring['min_price'] = min_price
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if max_price:
        querystring['max_price'] = max_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bluecart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(search_term: str, type: str, facets: str=None, delivery_type: str=None, category_id: str=None, rating: str=None, url: str=None, sort_by: str=None, page: str=None, max_price: str=None, min_price: str=None, condition: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Search Parameters are applicable when making a request with `type=search` to retrieve search results from Walmart. The search term is specified in the search_term parameter. The parameters should be appended as querystring parameters to the Walmart Product Data API GET HTTP request.
		
		Search results are retrieved from the Search results page on Walmart."
    search_term: A search term to use to search for Walmart items.
Either the search_term or url parameter must be supplied.
        facets: A free-form set of filter facets to use with the request. Available facet data is returned from `type=search` requests or can be retrieved from any Walmart URL in the `facet` Walmart URL querystring parameter.

Facet data should be expressed in comma seperated notation, with the name and value of the facet delimited by an equals `(\"=\")` character.

For example, to set the `brand` facet to the value `Garmin` and the `screen_size` facet to the value `6\"` the value of the facets request parameter would be `facets=brand=Garmin,screen_size=\"6\"`
        delivery_type: Determines whether results are filtered to just those matching the given delivery type. Valid values are:
pickup
Filter results to products available for pickup in store.
delivery_from_store
Filter results to products available delivery from a store.
shipping
Filter results to products available via home shipping.
available_in_store
Filter results to products available in store (use the customer_zipcode parameter to set the store location).
        category_id: 
Limits the search results returned from type=search requests to the category area specified in the search category_id property.

View the official Search Category IDs to get category_id values for use with Search Requests. Search Category IDs can also be retrieved from any Walmart URL in the cat_id Walmart URL querystring parameter.

Either the category_id or search_term parameter must be supplied when making type=search requests.

Note the category_id values supplied to a type=search request are Search Category IDs. They are not the same as the category_id values supplied to a type=category request and returned by the Categories API.
        rating: Determines whether results are filtered to just those matching the given customer star rating. Valid values are:
one_star
Filter results to products with a 1 to 1.9 star customer rating.
two_star
Filter results to products with a 2 to 2.9 star customer rating.
three_star
Filter results to products with a 3 to 3.9 star customer rating.
four_star
Filter results to products with a 4 to 5 star customer rating.
        url: The Walmart search results page to retrieve results from.

**Note: If the url parameter is supplied then all other parameters are ignored as they are overriden be those specified in the url parameter**
        sort_by: Determines how the results are sorted. Valid values are:

- best_seller
- price_high_to_low
- price_low_to_high
- best_match
- highest_rating
- newly_listed
- new
        page: The current page of results to retrieve. Inspect the pagination results property for details on the number of pages available.

update value of num to get additional pages
        max_price: Determines the maximum price of results to return, expressed in dollars and cents (i.e. min_price=39.99 for $39.99).
        min_price: 
Determines the minimum price of results to return, expressed in dollars and cents (i.e. min_price=2.99 for $2.99).
        condition: 
Filters the results to just products matching the given condition. Valid values are:
new
Filter results to just new products.
refurbished
Filter results to just refurbished products.
        
    """
    url = f"https://bluecart.p.rapidapi.com/request"
    querystring = {'search_term': search_term, 'type': type, }
    if facets:
        querystring['facets'] = facets
    if delivery_type:
        querystring['delivery_type'] = delivery_type
    if category_id:
        querystring['category_id'] = category_id
    if rating:
        querystring['rating'] = rating
    if url:
        querystring['url'] = url
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if condition:
        querystring['condition'] = condition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bluecart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offers(type: str, item_id: str='782866746', url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Offers Parameters are applicable when making a request with `type=offers` to retrieve seller Offers for a single product on Walmart - the product is specified using either the item_id parameter or the url parameter (where the url parameter contains a link to an Walmart offers page). The parameters should be appended as querystring parameters to the Walmart Product Data APIGETHTTP request.
		
		Offers are retrieved from the offers listing for a single product on Walmart."
    item_id: The Walmart Item ID (product ID) to retrieve Offers for. You can obtain item_id's from other API requests such as type=product and type=search

Note: If the item_id parameter is supplied then the url parameter is ignored.
        url: The Walmart offers-page URL to retrieve Offers from.

        
    """
    url = f"https://bluecart.p.rapidapi.com/request"
    querystring = {'type': type, }
    if item_id:
        querystring['item_id'] = item_id
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bluecart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(search_term: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When making a request with the type parameter set to type=autocomplete BlueCart API will return autocomplete suggestions for the search term specified in the search_term parameter."
    search_term: The search term to get autocomplete suggestions for.

        
    """
    url = f"https://bluecart.p.rapidapi.com/request"
    querystring = {'search_term': search_term, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bluecart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_profile(type: str, url: str=None, seller_id: str='10087', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Seller Profile Parameters are applicable when making a request with type=seller_profile to retrieve seller profile data for a single seller on Walmart - the seller is specified using either the seller_id or url parameter (where the url parameter contains a link to a Walmart seller page). The parameters should be appended as querystring parameters to the Walmart Product Data APIGETHTTP request.
		
		Seller Profile data is retrieved from the seller profile page for a single seller on Walmart."
    url: The Walmart seller-profile page URL to retrieve the seller profile from.

        seller_id: The Walmart Seller ID to retrieve seller data for.

**Note: If the seller_id parameter is supplied then the url parameter is ignored.**
        
    """
    url = f"https://bluecart.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if seller_id:
        querystring['seller_id'] = seller_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bluecart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews(type: str, walmart_domain: str=None, sort_by: str=None, url: str=None, item_id: str='782866746', page: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Reviews Parameters are applicable when making a request with type=reviews to retrieve customer Reviews for a single product on Walmart USA (walmart.com) or Walmart Canada (walmart.ca) - the product is specified using either the item_id or url parameter (where the url parameter contains a link to a Walmart product page). The parameters should be appended as querystring parameters to the Walmart Product Data APIGETHTTP request.
		
		Reviews are retrieved from the customer reviews page for a single product on Walmart."
    walmart_domain: The Walmart domain to retrieve Reviews for. Can be set to walmart.com (the default), or walmart.ca.
        sort_by: Determines the sort order of reviews to return. Valid values are:
most_relevant
Returns reviews with the most relevant reviews first.
most_helpful
Returns reviews with the most helpful reviews first.
newest_to_oldest
Returns reviews ordered from newest to oldest.
oldest_to_newest
Returns reviews ordered from oldest to newest.
high_to_low_rating
Returns reviews ordered highest to lowest rating.
low_to_high_rating
Returns reviews ordered lowest to highest rating.
        url: The Walmart reviews-page URL to retrieve Reviews from.

        item_id: 
The Walmart Item ID to retrieve Reviews for.

Note: If the item_id parameter is supplied then the url parameter is ignored.
        page: The current page of reviews to retrieve. Inspect the pagination.total_pages property in the Reviews results to see how many pages of reviews are available.

update value of num to get additional pages
        
    """
    url = f"https://bluecart.p.rapidapi.com/request"
    querystring = {'type': type, }
    if walmart_domain:
        querystring['walmart_domain'] = walmart_domain
    if sort_by:
        querystring['sort_by'] = sort_by
    if url:
        querystring['url'] = url
    if item_id:
        querystring['item_id'] = item_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bluecart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product(type: str, url: str=None, gtin: str=None, item_id: str='782866746', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Product Parameters are applicable when making a request with type=product to retrieve details of a single product on Walmart - the product is specified using either the item_id parameter or the url parameter (where the url parameter contains a link to a Walmart product page). The parameters should be appended as querystring parameters to the Walmart Product Data APIGETHTTP request.
		
		Product details are retrieved from the product page for a single product on Walmart."
    url: The Walmart product-page URL to retrieve product details from.

**Note: If the url parameter is supplied then the item_id and gtin parameters are ignored (the url overrides other parameters).**
        gtin: The GTIN/ISBN/UPC code to look up a matching product on Walmart for. If the gtin parameter is supplied then the item_id parameter is ignored. GTIN-based requests work by looking up the GTIN/ISBN/UPC on Walmart first, then retrieving the product details for the first matching item_id.
        item_id: The Walmart Item ID to retrieve product details for.

        
    """
    url = f"https://bluecart.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if gtin:
        querystring['gtin'] = gtin
    if item_id:
        querystring['item_id'] = item_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bluecart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

