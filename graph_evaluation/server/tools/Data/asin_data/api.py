import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def category(type: str, refinements: str=None, amazon_domain: str=None, page: int=None, sort_by: str=None, category_id: str=None, url: str='https://www.amazon.com/s?bbn=16225009011&rh=n%3A%2116225009011%2Cn%3A502394%2Cn%3A281052', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Category Parameters are applicable when making a request with `type=category` to retrieve category results for an Amazon category. Categories can be specified either via passing an Amazon category URL in to the url parameter, or by passing a Category ID in the `category_id` and an Amazon domain in the `amazon_domain` parameter. The parameters should be appended as querystring parameters to the Product Data API GET HTTP request.
		
		Category results are retrieved from the category listing page on Amazon."
    refinements: 
A comma-seperated list of refinement values to filter the category results by. These allow you to refine your category results by values such as \\\\\\\\\\\\\\\"Reviews rating 4 and over\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\" and \\\\\\\\\\\\\\\"brand\\\\\\\\\\\\\\\".

Refinement values are returned in the refinements array of each type=category result. Refinement values are dynamic and change by category area or search term used. If you wish to use refinements you should first issue a type=category request without specifying any refinements to retrieve a master list of the available refinements for the given category area/search term. You can then cache these refinement values for use on subsequent requests.

For example, to run a type=search request specifying two refinements with values `p_85/2470955011` and `p_36/2421886011` the value of the refinements parameter would be `refinements=p_85/2470955011,p_36/2421886011`

Note that sometimes Amazon do not present a explicit refinement value and instead a link is returned. In this instance you should pass the link into the url request parameter of your type=category request to retrieve data from that refinement-filtered page.
        amazon_domain: The Amazon domain to retrieve category results from. For a full list of supported Amazon domains see [Supported Amazon Domains](https://www.asindataapi.com/docs/product-data-api/reference/amazon-domains).
        page: The current page of category results to retrieve. Inspect the pagination.total_pages property in the Category results to see how many pages of category results are available.

update value of num to get additional pages
        sort_by: Determines the sort order of category results to return. Valid values are:

- most_recent
- price_low_to_high
- price_high_to_low
- featured
- average_review

        category_id: A category ID to retrieve results from. You may supply any arbitary value in the category_id parameter.

ASIN Data API will use the category_id in the following form: `https://amazon_domain/s?node=category_id.`

Note that pagination for top-level categories is not supported by the Amazon sites. If you wish to iterate the contents of a category the recommended approach is to pick the lowest level categories to perform your iteration / pagination on.
        url: The Amazon category results page URL to retrieve category results from. Be sure to URL-encode the url parameter.

**Note the url parameter is supplied, the category_id parameter cannot be used (as the url itself defines the category ID used).**
        
    """
    url = f"https://asin-data.p.rapidapi.com/request"
    querystring = {'type': type, }
    if refinements:
        querystring['refinements'] = refinements
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if category_id:
        querystring['category_id'] = category_id
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asin-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offers(type: str, offers_condition_used_acceptable: bool=None, offers_condition_used_good: bool=None, show_different_asins: bool=None, amazon_domain: str='amazon.com', offers_condition_used_like_new: bool=None, offers_free_shipping: bool=None, offers_condition_new: bool=None, page: int=None, asin: str='B073JYC4XM', url: str=None, offers_prime: bool=None, offers_condition_used_very_good: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Offers Parameters are applicable when making a request with `type=offers` to retrieve seller Offers for a single product on Amazon - the product is specified using either the asin and `amazon_domain` parameters or the url parameter (where the url parameter contains a link to an Amazon product offers page). The parameters should be appended as querystring parameters to the Product Data API GET HTTP request.
		
		Offers are retrieved from the offers listing popup window for a single product on Amazon."
    offers_condition_used_acceptable: 
Limit the offers returned to just those that are of Used-Acceptable Condition. Valid values are:
true
Only include offers that are Used-Acceptable Condition.
false
Include all offers, regardless of their Condition.
        offers_condition_used_good: 
Limit the offers returned to just those that are of Used-Good Condition. Valid values are:
true
Only include offers that are Used-Good Condition.
false
Include all offers, regardless of their Condition.
        show_different_asins: 
Sometimes Amazon will return Offers from ASINs other than the ASIN supplied in the asin request parameter (for example, when the original ASIN is out of stock). show_different_asins controls whether you want these other-ASIN offer results returned, or not. Can be set to true to include offers from other ASINs, or false (the default) to hide offers from ASINs other than the ASIN supplied in the asin parameter. Note that if you supply a url instead of asin in your request this parameter is ignored.
        amazon_domain: The Amazon domain to retrieve Offers for the product specified in the asin parameter from. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        offers_condition_used_like_new: 
Limit the offers returned to just those that are of Used-Like-New Condition. Valid values are:
true
Only include offers that are Used-Like-New Condition.
false
Include all offers, regardless of their Condition.
        offers_free_shipping: 
Limit the offers returned to just those that have Free Shipping. Valid values are:
true
Only include offers that have Free Shipping.
false
Include all offers, regardless of whether they have Free Shipping or not.
        offers_condition_new: l
Limit the offers returned to just those that are of New Condition. Valid values are:
true
Only include offers that are New Condition.
false
Include all offers, regardless of their Condition.
        page: The current page of reviews to retrieve. Inspect the pagination.total_pages property in the Reviews results to see how many pages of reviews are available.
**Note that the maximum number of reviews pages served by Amazon is 500**

update value of num to get additional pages
        asin: The Amazon ASIN (product ID) to retrieve Offers for. Used in combination with the amazon_domain parameter.

**Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.**
        url: The Amazon product-page URL to retrieve Offers from.

**Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignored.**
        offers_prime: 
Limit the offers returned to just those that are Prime-eligible. Valid values are:
true
Only include offers that are Prime-eligible.
false
Include all offers, regardless of whether they are Prime-eligible or not.
        offers_condition_used_very_good: 
Limit the offers returned to just those that are of Used-Very-Good Condition. Valid values are:
true
Only include offers that are Used-Very-Good Condition.
false
Include all offers, regardless of their Condition.
        
    """
    url = f"https://asin-data.p.rapidapi.com/request"
    querystring = {'type': type, }
    if offers_condition_used_acceptable:
        querystring['offers_condition_used_acceptable'] = offers_condition_used_acceptable
    if offers_condition_used_good:
        querystring['offers_condition_used_good'] = offers_condition_used_good
    if show_different_asins:
        querystring['show_different_asins'] = show_different_asins
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if offers_condition_used_like_new:
        querystring['offers_condition_used_like_new'] = offers_condition_used_like_new
    if offers_free_shipping:
        querystring['offers_free_shipping'] = offers_free_shipping
    if offers_condition_new:
        querystring['offers_condition_new'] = offers_condition_new
    if page:
        querystring['page'] = page
    if asin:
        querystring['asin'] = asin
    if url:
        querystring['url'] = url
    if offers_prime:
        querystring['offers_prime'] = offers_prime
    if offers_condition_used_very_good:
        querystring['offers_condition_used_very_good'] = offers_condition_used_very_good
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asin-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews(type: str, review_stars: str=None, page: int=None, search_term: str=None, show_different_asins: bool=None, review_media_type: str=None, sort_by: str=None, global_reviews: bool=None, review_id: str=None, reviewer_type: str=None, amazon_domain: str='amazon.com', url: str=None, review_formats: str=None, asin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Reviews Parameters are applicable when making a request with `type=reviews` to retrieve customer Reviews for a single product on Amazon - the product is specified using either the asin and `amazon_domain` parameters or the url parameter (where the url parameter contains a link to an Amazon reviews page). The parameters should be appended as querystring parameters to the Product Data API GET HTTP request.
		
		Reviews are retrieved from the customer reviews page for a single product on Amazon."
    review_stars: The star rating of reviews to retrieve. Valid values are:

- all_stars 
- five_star
- four_star
- three_star
- two_star
- one_star
- all_positive
- all_critical
        page: The current page of reviews to retrieve. Inspect the pagination.total_pages property in the Reviews results to see how many pages of reviews are available.

**Note that the maximum number of reviews pages served by Amazon is 500.**

update value of num to get additional pages
        search_term: A search term to use to search reviews.
        show_different_asins: Sometimes Amazon will return Reviews from ASINs other than the ASIN supplied in the asin request parameter (for example, when the original ASIN is out of stock). `show_different_asins `controls whether you want these other-ASIN review results returned, or not. Can be set to true (the default) to include reviews from other ASINs, or false to hide reviews from ASINs other than the ASIN supplied in the asin parameter. 

Note that if you supply a url instead of asin in your request this parameter is ignored.

Note in the event of` show_different_asins=false` and the ASIN shown by Amazon being different from the requested ASIN, then all fields apart from product will be removed from the response.
        review_media_type: Filter the reviews to those containing a specific media type. Valid values are:

`all_reviews`
- Include reviews with text, images or video.

`media_reviews_only`
- Include only reviews containing images or video.
        sort_by: Determines the sort order of reviews to return. 

Valid values are:  `most_helpful` and `most_recent`

        global_reviews: Determines whether Global Reviews are included, or not Global Reviews are reviews from other Amazon domains outside of the Amazon domain specified in the request. Valid values are:

`true`
- The  default, Global Reviews are included in the results.

`false`
- Global reviews are excluded from the results.
        review_id: 
A single Review ID, as returned by a prior `type=reviews` request, to retrieve. Useful if you wish to check for the presence of a single review. Use in combination with the amazon_domain parameter.
        reviewer_type: The type of reviewer to retrieve reviews from. Valid values are:

`verified_purchase`
- include reviews by Amazon Verified Purchasers only

`all`
- include  all reviews, regardless of whether they are from Amazon Verified Purchasers or not)
        amazon_domain: The Amazon domain to retrieve Reviews for the product specified in the asin parameter from. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        url: The Amazon product-page URL to retrieve Reviews from.

**Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignored.**
        review_formats: The type of reviewer to retrieve reviews from. Valid values are:

`all_formats`
- Include reviews of any product format/variant.

`current_format`
- Include reviews relating specifically to the current format/variant.
        asin: The Amazon ASIN (product ID) to retrieve Reviews for. Used in combination with the amazon_domain parameter.

**Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.**
        
    """
    url = f"https://asin-data.p.rapidapi.com/request"
    querystring = {'type': type, }
    if review_stars:
        querystring['review_stars'] = review_stars
    if page:
        querystring['page'] = page
    if search_term:
        querystring['search_term'] = search_term
    if show_different_asins:
        querystring['show_different_asins'] = show_different_asins
    if review_media_type:
        querystring['review_media_type'] = review_media_type
    if sort_by:
        querystring['sort_by'] = sort_by
    if global_reviews:
        querystring['global_reviews'] = global_reviews
    if review_id:
        querystring['review_id'] = review_id
    if reviewer_type:
        querystring['reviewer_type'] = reviewer_type
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if url:
        querystring['url'] = url
    if review_formats:
        querystring['review_formats'] = review_formats
    if asin:
        querystring['asin'] = asin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asin-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(type: str, search_term: str='memory cards', url: str=None, direct_search: str=None, page: int=None, exclude_sponsored: bool=None, sort_by: str=None, refinements: str=None, category_id: str=None, amazon_domain: str='amazon.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Search Parameters are applicable when making a request with `type=search` to retrieve search results for an Amazon domain - the Amazon domain is specified using the `amazon_domain` parameter and the search term is specified in the `search_term` parameter. The parameters should be appended as querystring parameters to the Product Data API GET HTTP request.
		
		Search results are retrieved from the search results page on Amazon."
    search_term: A search term to use to search products.
        url: The Amazon search results page URL to retrieve search results from. Be sure to URL-encode the url parameter.

**Note the url parameter is supplied, the search_term parameter cannot be used (as the url itself defines the search term used).**
        direct_search: By default Amazon will, if a spelling mistake is suspected in the search_term, show search results for the corrected search term.

To disable this behaviour and search for the value in `search_term` directly, without auto-correcting it, set `direct_search=true`.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        exclude_sponsored: Whether to exclude sponsored search results (when set to exclude_sponsored=true) from the search results returned, or not. Defaults to false.
        sort_by: Determines the sort order of search results to return. Valid values are:

- most_recent
- price_low_to_high
- price_high_to_low
- featured
- average_review
        refinements: 
A comma-seperated list of refinement values to filter the search results by. These allow you to refine your search by values such as \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"Reviews rating 4 and over\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" and \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"brand\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".

Refinement values are returned in the refinements array of each type=search result. Refinement values are dynamic and change by category area or search term used. If you wish to use refinements you should first issue a type=search request without specifying any refinements to retrieve a master list of the avaialble refinements for the given category area/search term. You can then cache these refinement values for use on subsequent requests.

For example, to run a type=search request specifying two refinements with values `p_85/2470955011 `and` p_36/2421886011 `the value of the refinements parameter would be `refinements=p_85/2470955011,p_36/2421886011`
        category_id: 
A category ID to limit search results to. ASIN Data API will use the category_id in the following form: https://amazon_domain/s?node=category_id.

Note that pagination for top-level categories is not supported by the Amazon sites. If you wish to iterate the contents of a category the recommended approach is to pick the lowest level categories to perform your iteration / pagination on.
        amazon_domain: The Amazon domain to retrieve search results from. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        
    """
    url = f"https://asin-data.p.rapidapi.com/request"
    querystring = {'type': type, }
    if search_term:
        querystring['search_term'] = search_term
    if url:
        querystring['url'] = url
    if direct_search:
        querystring['direct_search'] = direct_search
    if page:
        querystring['page'] = page
    if exclude_sponsored:
        querystring['exclude_sponsored'] = exclude_sponsored
    if sort_by:
        querystring['sort_by'] = sort_by
    if refinements:
        querystring['refinements'] = refinements
    if category_id:
        querystring['category_id'] = category_id
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asin-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product(type: str, gtin: str=None, url: str=None, amazon_domain: str='amazon.com', asin: str='B073JYC4XM', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Product Parameters are applicable when making a request with `type=product` to retrieve details of a single product on Amazon - the product is specified using either the asin and `amazon_domain` parameters or the url parameter (where the url parameter contains a link to an Amazon product page). The parameters should be appended as querystring parameters to the Product Data API GET HTTP request.
		
		Product details are retrieved from the product page for a single product on Amazon."
    gtin: A GTIN, ISBN, UPC or EAN code to retrieve results for. Internally ASIN Data API will first convert the GTIN/ISBN/UPC/EAN to an Amazon ASIN, then retrieve the results for that ASIN from Amazon. Used in combination with the amazon_domain parameter.

**Note: If the gtin and amazon_domain parameters are supplied then the url parameter is ignored.**
        url: The Amazon product-page URL to retrieve product details from.

**Ensure that the url parameter is URL-encoded.**

Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignored.
        amazon_domain: The Amazon domain to retrieve product details from for the product specified in the asin parameter. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        asin: The Amazon ASIN (product ID) to retrieve product details for. Used in combination with the `amazon_domain` parameter.

**Note: If the asin and `amazon_domain `parameters are supplied then the url parameter is ignored.**
        
    """
    url = f"https://asin-data.p.rapidapi.com/request"
    querystring = {'type': type, }
    if gtin:
        querystring['gtin'] = gtin
    if url:
        querystring['url'] = url
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if asin:
        querystring['asin'] = asin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asin-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

