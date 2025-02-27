import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product(type: str, gtin: str='317061059', url: str=None, output: str=None, item_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Product Parameters are applicable when making a request with `type=product` to retrieve details of a single product on Home Depot - the product is specified using either the item_id parameter or the url parameter (where the url parameter contains a link to a Home Depot product page). The parameters should be appended as querystring parameters to the Home Depot Product Data APIGETHTTP request.
		
		Product details are retrieved from the product page for a single product on Home Depot"
    gtin: The GTIN/ISBN/UPC code to look up a matching product on Home Depot for. If the gtin parameter is supplied then the item_id parameter is ignored. GTIN-based requests work by looking up the GTIN/ISBN/UPC on Home Depot first, then retrieving the product details for the first matching item_id.
        url: The Home Depot product-page URL to retrieve product details from.

Use either `item_id` OR `url`. 

If the item_id is included, do **NOT** include the URL

**Note: If the url parameter is supplied then the item_id and gtin parameters are ignored (the url overrides other parameters).**
        output: Determines the format in which results are returned: `JSON`, `HTML` or `CSV`

JSON is the default 
        item_id: The Home Depot Item ID to retrieve product details for.
        
    """
    url = f"https://bigbox.p.rapidapi.com/request"
    querystring = {'type': type, }
    if gtin:
        querystring['gtin'] = gtin
    if url:
        querystring['url'] = url
    if output:
        querystring['output'] = output
    if item_id:
        querystring['item_id'] = item_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews(item_id: int, type: str, page: str=None, output: str=None, four_star: bool=None, sort_by: str=None, reviewer_type: str=None, gtin: str=None, two_star: bool=None, five_star: bool=None, one_star: bool=None, three_star: bool=None, search_term: str=None, url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Reviews Parameters are applicable when making a request with `type=reviews` to retrieve customer Reviews for a single product on Home Depot - the product is specified using either the item_id or url parameter (where the url parameter contains a link to a Home Depot product page). The parameters should be appended as querystring parameters to the Home Depot Product Data API GET HTTP request.
		
		Reviews are retrieved from the customer reviews page for a single product on Home Depot."
    item_id: The Home Depot Item ID to retrieve Reviews for.

**Note: If the item_id parameter is supplied then the url parameter is ignored.**
        page: The current page of reviews to retrieve. Inspect the pagination.total_pages property in the Reviews results to see how many pages of reviews are available.

**Note the maximum number of reviews per page is 10 (this is a Home Depot limit).**

update value of num to get additional pages
        output: Determines the format in which results are returned: `JSON`, `HTML` or `CSV`

JSON is the default 
        four_star: Set to true to filter to only reviews that give the product a 4-star rating.
        sort_by: Determines the sort order of reviews to return. Valid values are:

- photo_reviews
- most_helpful
- newest_to_oldest
- oldest_to_newest
- high_to_low_rating
- low_to_high_rating
        reviewer_type: The type of reviewer to retrieve reviews from. Valid values are:

`verified_purchase` : Include reviews by Verified Purchasers only.
`all` : Include all reviews, regardless of whether they are from Verified Purchasers or not.
        gtin: The GTIN/ISBN/UPC code to look up a matching product on Home Depot for to retrieve reviews for. If the gtin parameter is supplied then the `item_id` parameter is ignored. GTIN-based requests work by looking up the GTIN/ISBN/UPC on Home Depot first, then retrieving the reviews for the first matching `item_id`.
        two_star: Set to true to filter to only reviews that give the product a 2-star rating.
        five_star: Set to true to filter to only reviews that give the product a 5-star rating.
        one_star: Set to true to filter to only reviews that give the product a 1-star rating.
        three_star: Set to true to filter to only reviews that give the product a 3-star rating.
        search_term: A search term to use to search reviews.
        url: The Home Depot reviews-page URL to retrieve Reviews from.

        
    """
    url = f"https://bigbox.p.rapidapi.com/request"
    querystring = {'item_id': item_id, 'type': type, }
    if page:
        querystring['page'] = page
    if output:
        querystring['output'] = output
    if four_star:
        querystring['four_star'] = four_star
    if sort_by:
        querystring['sort_by'] = sort_by
    if reviewer_type:
        querystring['reviewer_type'] = reviewer_type
    if gtin:
        querystring['gtin'] = gtin
    if two_star:
        querystring['two_star'] = two_star
    if five_star:
        querystring['five_star'] = five_star
    if one_star:
        querystring['one_star'] = one_star
    if three_star:
        querystring['three_star'] = three_star
    if search_term:
        querystring['search_term'] = search_term
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def questions(type: str, output: str=None, page: str=None, sort_by: str=None, gtin: str=None, item_id: int=317061059, url: str=None, search_term: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Questions Parameters are applicable when making a request with type=questions to retrieve customer Questions for a single product on Home Depot - the product is specified using either the item_id or url parameter (where the url parameter contains a link to a Home Depot product page). The parameters should be appended as querystring parameters to the Home Depot Product Data APIGETHTTP request.
		
		Questions are retrieved from the customer questions page for a single product on Home Depot."
    output: Determines the format in which results are returned: `JSON`, `HTML` or `CSV`

JSON is the default 
        page: The current page of questions to retrieve. Inspect the pagination.total_pages property in the Questions results to see how many pages of questions are available.

**Note the maximum number of questions per page is 10 (this is a Home Depot limit).**

update value of num to get additional pages
        sort_by: Determines the sort order of questions to return. Valid values are:

- newest_to_oldest
- oldest_to_newest
- most_answered
- most_helpful
- featured_questions
- can_you_answer
- filter first [questions with no answer yet]
        gtin: The GTIN/ISBN/UPC code to look up a matching product on Home Depot for to retrieve questions for. If the gtin parameter is supplied then the `item_id `parameter is ignored. GTIN-based requests work by looking up the GTIN/ISBN/UPC on Home Depot first, then retrieving the questions for the first matching `item_id`
        item_id: The Home Depot Item ID to retrieve Questions for.

**Note: If the item_id parameter is supplied then the url parameter is ignored.**
        url: The Home Depot questions-page URL to retrieve Questions from.
        search_term: A search term to use to search questions.
        
    """
    url = f"https://bigbox.p.rapidapi.com/request"
    querystring = {'type': type, }
    if output:
        querystring['output'] = output
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if gtin:
        querystring['gtin'] = gtin
    if item_id:
        querystring['item_id'] = item_id
    if url:
        querystring['url'] = url
    if search_term:
        querystring['search_term'] = search_term
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category(category_id: str, type: str, sort_by: str=None, output: str=None, max_price: str=None, min_price: str=None, page: str='2', url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Category Parameters are applicable when making a request with `type=category` to retrieve category results from Home Depot. The category is specified in the category_id parameter. The parameters should be appended as querystring parameters to the Home Depot Product Data API GET HTTP request.
		
		Category results are retrieved from the Category results page on Home Depot."
    category_id: A category ID to retrieve results from. You may supply any arbitary value in the category_id parameter however we recommend using a category ID returned from the Categories API as these are known-good category IDs from Home Depot.
        sort_by: Determines how the results are sorted. Valid values are:

- best_seller
- most_popular
- price_high_to_low
- price_low_to_high
- highest_rating
        output: Determines the format in which results are returned: `JSON`, `HTML` or `CSV`

JSON is the default 
        max_price: 
Determines the maximum price of results to return, expressed in dollars and cents (i.e. `min_price=39.99` for $39.99).
        min_price: 
Determines the maximum price of results to return, expressed in dollars and cents (i.e. `min_price=39.99` for $39.99).
        page: The current page of results to retrieve. Inspect the pagination results property for details on the number of pages available.

update value of num to get additional pages
        url: The Home Depot category results page to retrieve results from.

**Note: If the url parameter is supplied then all other parameters are ignored as they are overriden be those specified in the url parameter.**
        
    """
    url = f"https://bigbox.p.rapidapi.com/request"
    querystring = {'category_id': category_id, 'type': type, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if output:
        querystring['output'] = output
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if page:
        querystring['page'] = page
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(search_term: str, type: str, output: str=None, max_price: str='max_price=500', sort_by: str=None, url: str=None, page: int=None, min_price: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Search Parameters are applicable when making a request with `type=search` to retrieve search results from Home Depot. The search term is specified in the search_term parameter. The parameters should be appended as querystring parameters to the Home Depot Product Data API GET HTTP request.
		
		Search results are retrieved from the Search results page on Home Depot."
    search_term: A search term to use to search for Home Depot items.
Either the search_term or url parameter must be supplied.
        output: Determines the format in which results are returned: `JSON`, `HTML` or `CSV`

JSON is the default 
        max_price: Determines the maximum price of results to return, expressed in dollars and cents (i.e. `max_price=39.99` for $39.99).
        sort_by: Determines how the results are sorted. Valid values are:

- best_seller
- most_popular
- price_high_to_low
- price_low_to_high
- highest_rating
        url: The Home Depot search results page to retrieve results from.

**Note: If the url parameter is supplied then all other parameters are ignored as they are overriden be those specified in the url parameter**
        page: The current page of results to retrieve. Inspect the pagination results property for details on the number of pages available.

update value of num to get additional pages
        min_price: Determines the minimum price of results to return, expressed in dollars and cents (i.e. `min_price=2.99` for $2.99).
        
    """
    url = f"https://bigbox.p.rapidapi.com/request"
    querystring = {'search_term': search_term, 'type': type, }
    if output:
        querystring['output'] = output
    if max_price:
        querystring['max_price'] = max_price
    if sort_by:
        querystring['sort_by'] = sort_by
    if url:
        querystring['url'] = url
    if page:
        querystring['page'] = page
    if min_price:
        querystring['min_price'] = min_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

