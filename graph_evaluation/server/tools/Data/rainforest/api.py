import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def author_page(type: str, url: str=None, format_id: str=None, amazon_domain: str='amazon.com', asin: str='B000APVZ7W', page: int=None, sort_by: str=None, customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Author Page Parameters are applicable when making a request with type=author_page to retrieve data from the authors page single author on Amazon - the author is specified using either the author asin and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon authors page). The parameters should be appended as querystring parameters to the Product Data API GET HTTP request.
		
		Author Page data is retrieved from the author page for a single author on Amazon."
    url: 
The Amazon author page URL to retrieve data from.

Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignored.
        format_id: 
A format ID to determine the format of author titles returned in the titles array. Format ID's are returned in the format_ids array in each type=author_page results (this is necessary are the format_id value is dynamic and can change over time).
        amazon_domain: The Amazon domain to retrieve author page data for the author specified in the asin parameter. For a full list of supported Amazon domains see Supported Amazon Domains.

Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.
        asin: 
The Amazon author ASIN to retrieve author page data for. Used in combination with the amazon_domain parameter.

Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        sort_by: Determines the sort order of author titles array that is returned. Valid values are:
popularity
Sort the author titles array by most popular first.
price_low_to_high
Sort the author titles array by lowest price first.
price_high_to_low
Sort the author titles array by highest price first.
average_review
Sort the author titles array by highest average review first.
publication_date
Sort the author titles array by most recent publication date first.
most_reviews
Sort the author titles array by those with the most reviews first.

        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \"France\" for requests for pages from \"amazon.fr\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if format_id:
        querystring['format_id'] = format_id
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if asin:
        querystring['asin'] = asin
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def also_bought(type: str, url: str=None, asin: str='B084ZF5D65', amazon_domain: str='amazon.com', customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Also Bought Parameters are applicable when making a request with type=also_bought to retrieve also bought details for a single product on Amazon - the product is specified using either the asin and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon product page). The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Also bought details are retrieved from the product page for a single product on Amazon. An type=also_bought request will automatically paginate through all of the pages of also-bought data and return every also bought result."
    url: The Amazon product page URL to retrieve also bought results from.

        asin: The Amazon ASIN (product ID) to retrieve also bought details for. Used in combination with the amazon_domain parameter.

Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.
        amazon_domain: 
The Amazon domain to retrieve also bought details from for the product specified in the asin parameter. For a full list of supported Amazon domains see Supported Amazon Domains.

Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \"France\" for requests for pages from \"amazon.fr\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if asin:
        querystring['asin'] = asin
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offers(type: str, offers_prime: bool=None, offers_condition_used_very_good: bool=None, offers_condition_used_acceptable: bool=None, offers_condition_new: bool=None, url: str=None, page: int=None, offers_condition_used_like_new: bool=None, amazon_domain: str='amazon.com', asin: str='B073JYC4XM', gtin: str=None, offers_condition_used_good: bool=None, offers_free_shipping: bool=None, customer_location: str=None, show_different_asins: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Offers are retrieved from the offers listing popup window for a single product on Amazon.
		
		The Offers Parameters are applicable when making a request with type=offers to retrieve seller Offers for a single product on Amazon - the product is specified using either the asin and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon product offers page). The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request."
    offers_prime: 
Limit the offers returned to just those that are Prime-eligible. Valid values are:
true
Only include offers that are Prime-eligible.
false
Include all offers, regardless of whether they are Prime-eligible or not.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        amazon_domain: The Amazon domain to retrieve Offers for the product specified in the asin parameter from. For a full list of supported Amazon domains see [Supported Amazon Domains.](https://www.rainforestapi.com/docs/product-data-api/reference/amazon-domains)

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        asin: The Amazon ASIN (product ID) to retrieve Offers for. Used in combination with the amazon_domain parameter.

**Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.**
        gtin: A GTIN, ISBN, UPC or EAN code to retrieve results for. Internally Rainforest will first convert the GTIN/ISBN/UPC/EAN to an Amazon ASIN, then retrieve the results for that ASIN from Amazon. Used in combination with the amazon_domain parameter.

Note: If the gtin and amazon_domain parameters are supplied then the url parameter is ignored.
        offers_free_shipping: 
Limit the offers returned to just those that have Free Shipping. Valid values are:
true
Only include offers that have Free Shipping.
false
Include all offers, regardless of whether they have Free Shipping or not.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. `France` for requests for pages from `amazon.fr`

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        show_different_asins: Sometimes Amazon will return Offers from ASINs other than the ASIN supplied in the asin request parameter (for example, when the original ASIN is out of stock). show_different_asins controls whether you want these other-ASIN offer results returned, or not. Can be set to true to include offers from other ASINs, or false (the default) to hide offers from ASINs other than the ASIN supplied in the asin parameter. Note that if you supply a url instead of asin in your request this parameter is ignored.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if offers_prime:
        querystring['offers_prime'] = offers_prime
    if offers_condition_used_very_good:
        querystring['offers_condition_used_very_good'] = offers_condition_used_very_good
    if offers_condition_used_acceptable:
        querystring['offers_condition_used_acceptable'] = offers_condition_used_acceptable
    if offers_condition_new:
        querystring['offers_condition_new'] = offers_condition_new
    if url:
        querystring['url'] = url
    if page:
        querystring['page'] = page
    if offers_condition_used_like_new:
        querystring['offers_condition_used_like_new'] = offers_condition_used_like_new
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if asin:
        querystring['asin'] = asin
    if gtin:
        querystring['gtin'] = gtin
    if offers_condition_used_good:
        querystring['offers_condition_used_good'] = offers_condition_used_good
    if offers_free_shipping:
        querystring['offers_free_shipping'] = offers_free_shipping
    if customer_location:
        querystring['customer_location'] = customer_location
    if show_different_asins:
        querystring['show_different_asins'] = show_different_asins
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def questions(type: str, asin: str='B073JYC4XM', sort_by: str=None, customer_location: str=None, amazon_domain: str='amazon.com', page: int=None, url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Questions Parameters are applicable when making a request with type=questions to retrieve customer Questions & Answers for a single product on Amazon - the product is specified using either the asin and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon product page). The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Questions are retrieved from the customer questions page for a single product on Amazon"
    asin: The Amazon ASIN (product ID) to retrieve Questions & Answers for. Used in combination with the amazon_domain parameter.

**Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.**
        sort_by: Determines the sort order of Questions & Answers to return. Valid values are:

most_helpful
Returns Questions & Answers with the most helpful first.
most_recent
Returns Questions & Answers in date order, with the most recent first.

        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \\\"France\\\" for requests for pages from \\\"amazon.fr\\\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        amazon_domain: The Amazon domain to retrieve Questions & Answers for the product specified in the asin parameter from. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        url: The Amazon product-page URL to retrieve Questions & Answers from.

**Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignored.**
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if asin:
        querystring['asin'] = asin
    if sort_by:
        querystring['sort_by'] = sort_by
    if customer_location:
        querystring['customer_location'] = customer_location
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if page:
        querystring['page'] = page
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deals(type: str, prime_eligible: bool=None, minimum_rating: str='1_and_up', deal_states: str=None, amazon_domain: str='amazon.com', category_id: str='2619525011', url: str=None, sort_by: str='price_high_to_low', deal_types: str='best_deal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Deals Parameters are applicable when making a request with type=deals to retrieve Lightning Deal / Daily Deals results for an Amazon lightning deals / daily deals / 'goldbox' page. The category of deals and Amazon domain are specified using the category_id and amazon_domain parameter. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Deals results are retrieved from the deals listing page on Amazon.
		
		You can retrieve category IDs for each amazon_domain using the Categories API. Category IDs are also returned in the categories array in each type=deals response.
		
		Multiple category IDs can be supplied, comma seperated, in the category_id parameter. For example, to request deals for Beauty (3760911) and Electronics (172282) categories on amazon.com the parameters would be category_id=3760911,172282&amazon_domain=amazon.com.
		
		If you wish to retrieve Deals from the deals homepage for a given amazon_domain, simply omit the category_id parameter entirely."
    minimum_rating: Only include deals for products with a customer rating meeting the specified criteria. If you omit the minimum_rating parameter deals for products with any rating are included. Valid values are:

1_and_up
Include deals for products with a 1-star customer rating and upwards.
2_and_up
Include deals for products with a 2-star customer rating and upwards.
3_and_up
Include deals for products with a 3-star customer rating and upwards.
4_and_up
Include deals for products with a 4-star customer rating and upwards.
        deal_states: The types of deal state to include. If you omit the deal_states parameter then all deal states are included.

**Note multiple deal states can be specified, comma-separated.**

Valid values are:
- available
- Include available-state deals.

upcoming
Include upcoming-state deals.
expired
Include expired-state deals.
        amazon_domain: The Amazon domain to retrieve Deals from for the category ID(s) specified in the category_id parameter. For a full list of supported Amazon domains see Supported Amazon Domains.
        category_id: A comma-seperated list of Deals category IDs to retrieve deals results from, as returned by the Categories API. Category IDs are also returned in the categories array in each type=deals response.

Multiple category IDs can be supplied, comma seperated, in the category_id parameter. For example, to request deals for Beauty (3760911) and Electronics (172282) categories on amazon.com the parameters would be category_id=3760911,172282&amazon_domain=amazon.com.

If you wish to retrieve Deals from the deals homepage for a given amazon_domain, simply omit the category_id parameter entirely.
        url: The Amazon deals page URL to retrieve deals results from. This parameter, as it specifies a URL, should itself be url encoded.

**Note using the url parameter for Deals requests is not recommended as due to the session and pagination characteristics of the Deals pages simply specifying the url can prove unreliable in all but the simplest use-cases. It is recommended that you use the category_id and amazon_domain parameters to specify your Deals request instead.**
        sort_by: Determines the sort order of the deals returned.

Valid values are:

featured
Sort by featured deals.
discount_low_to_high
Sort by discount, lowest discount first.
discount_high_to_low
Sort by discount, highest discount first.
price_low_to_high
Sort by price, lowest price first.
price_high_to_low
Sort by price, highest price first.
        deal_types: The types of deal to include. If you omit the deal_types parameter then all deal types are included.

**Note multiple deal types can be specified, comma-separated.**
Valid values are:

deal_of_the_day
Include deal-of-the-day deals.
lightning_deal
Include lightning deals.
best_deal
Include best-deal deals.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if prime_eligible:
        querystring['prime_eligible'] = prime_eligible
    if minimum_rating:
        querystring['minimum_rating'] = minimum_rating
    if deal_states:
        querystring['deal_states'] = deal_states
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if category_id:
        querystring['category_id'] = category_id
    if url:
        querystring['url'] = url
    if sort_by:
        querystring['sort_by'] = sort_by
    if deal_types:
        querystring['deal_types'] = deal_types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autcomplete(search_term: str, type: str, amazon_domain: str, autcomplete_alias: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When making a request with the type parameter set to type=autocomplete Rainforest API will return autocomplete suggestions for the search term specified in the search_term parameter, on the Amazon domain specified in the amazon_domain parameter.
		You may scope your autocomplete request to an Amazon autocomplete alias (synonymous with a product area or top-level category) by specifying the alias using the autocomplete_alias parameter. The available autocomplete_alias values vary depending on the amazon_domain used. You can find a full list of all autocomplete aliaes, per Amazon-domain, in the autocomplete aliases reference."
    search_term: The search term to get autocomplete suggestions for. Used in combination with the amazon_domain parameter.
        type: The Amazon domain to retrieve autocomplete details from for the search term specified in the search_term parameter. For a full list of supported Amazon domains see Supported Amazon Domains.
        autcomplete_alias: 
The alias to use to scope the autocomplete request. Autocomplete aliases can be thought of as the \"category\" area that the autocomplete request runs in. If none is specified then Rainforest will default to running the autocomplete request in the top-level \"All Departments\" autocomplete alias.

For a full list of supported Autocomplete Aliases see Supported Autocomplete Aliases.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'search_term': search_term, 'type': type, 'amazon_domain': amazon_domain, }
    if autcomplete_alias:
        querystring['autcomplete_alias'] = autcomplete_alias
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asin_to_gtin(amazon_domain: str, type: str, asin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When making a request with the type parameter set to type=asin_to_gtin Rainforest API will attempt to look up GTIN / EAN / UPC / ISBN numbers for the ASIN and Amazon domain specified in the asin and amazon_domain parameters."
    amazon_domain: The Amazon domain to retrieve ASIN to GTIN details from for the ASIN specified in the asin parameter. For a full list of supported Amazon domains see Supported Amazon Domains.
        asin: The Amazon ASIN (product ID) to retrieve GTIN(s) for. Used in combination with the amazon_domain parameter.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'amazon_domain': amazon_domain, 'type': type, 'asin': asin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def formats_editions(type: str, amazon_domain: str=None, url: str=None, asin: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Formats & Editions Parameters are applicable when making a request with the type parameter set to type=formats_editions Rainforest API will return data from the Formats & Editions popup (typically shown on books and collectables product pages to list all of the ASINs of different formats of the current ASIN) for the product specified in either the asin and amazon_domain parameters or the url parameter. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Formats & Editions data is retrieved from the Formats & Editions popup for a single product on Amazon"
    amazon_domain: 
The Amazon domain to retrieve Formats & Editions for the product specified in the asin parameter from. For a full list of supported Amazon domains see Supported Amazon Domains.

Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.
        url: The Amazon product-page URL to retrieve Formats & Editions from.

Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignor
        asin: The Amazon ASIN (product ID) to retrieve Formats & Editions for. Used in combination with the amazon_domain parameter.

Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if url:
        querystring['url'] = url
    if asin:
        querystring['asin'] = asin
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def question_answers(type: str, question_id: str='Tx2YPKCGKAORFA4', page: int=None, amazon_domain: str='amazon.com', customer_location: str=None, url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Question Answers Parameters are applicable when making a request with type=question_answers to retrieve all answers for a single question on Amazon - the question is specified using either the question_id and amazon_domain parameters or the url parameter (where the url parameter contains a link to a question answers page). The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Question Answers are retrieved from the customer question answers page for a single question on Amazon."
    question_id: The ID of the question to retrieve answers for.

Rainforest API returns question_id's from `type=questions` requests in the id parameter. See the questions parameters for further information.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        amazon_domain: The Amazon domain to retrieve answers for the question specified in the question_id parameter from. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and question_id parameters are supplied then the url parameter is ignored**.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \\\"France\\\" for requests for pages from \\\"amazon.fr\\\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        url: The Amazon product-page URL to retrieve Question Answers from.

**Note: If the url parameter is supplied then the amazon_domain and question_id parameters are ignored.**
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if question_id:
        querystring['question_id'] = question_id
    if page:
        querystring['page'] = page
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if customer_location:
        querystring['customer_location'] = customer_location
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews(type: str, url: str=None, global_reviews: bool=None, review_stars: str=None, sort_by: str=None, search_term: str=None, page: int=None, reviewer_type: str=None, review_media_type: str=None, customer_location: str=None, gtin: str=None, asin: str='B073JYC4XM', amazon_domain: str='amazon.com', review_format: str=None, show_different_asins: bool=None, review_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Reviews Parameters are applicable when making a request with type=reviews to retrieve customer Reviews for a single product on Amazon - the product is specified using either the asin and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon reviews page). The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Reviews are retrieved from the customer reviews page for a single product on Amazon."
    url: The Amazon product-page URL to retrieve Reviews from.

**Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignored.**
        global_reviews: Determines whether Global Reviews are included, or not Global Reviews are reviews from other Amazon domains outside of the Amazon domain specified in the request. Valid values are:

true
The default, Global Reviews are included in the results.
false
Global reviews are excluded from the results.
        review_stars: The star rating of reviews to retrieve. Valid values are:

- all_stars
- five_star
- four_star
- three_star
- two_star
- one_star
- all_positive
- all_critical
        sort_by: Determines the sort order of reviews to return. Valid values are:

- most_helpful
- most_recent
        search_term: A search term to use to search reviews.
        page: The current page of search results to retrieve. Inspect the `pagination.total_pages` property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        reviewer_type: The type of reviewer to retrieve reviews from. Valid values are:

`verified_purchase`
Include reviews by Amazon Verified Purchasers only.
`all`
Include all reviews, regardless of whether they are from Amazon Verified Purchasers or not.
        review_media_type: Filter the reviews to those containing a specific media type. Valid values are:

all_reviews
Include reviews with text, images or video.
media_reviews_only
Include only reviews containing images or video.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \\\"France\\\" for requests for pages from \\\"amazon.fr\\\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.
        gtin: A GTIN, ISBN, UPC or EAN code to retrieve results for. Internally Rainforest will first convert the GTIN/ISBN/UPC/EAN to an Amazon ASIN, then retrieve the results for that ASIN from Amazon. Used in combination with the amazon_domain parameter.

**Note: If the gtin and amazon_domain parameters are supplied then the url parameter is ignored.**
        asin: The Amazon ASIN (product ID) to retrieve Reviews for. Used in combination with the amazon_domain parameter.

**Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.**
        amazon_domain: The Amazon domain to retrieve Reviews for the product specified in the asin parameter from. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        review_format: he type of reviewer to retrieve reviews from. Valid values are:

all_formats
Include reviews of any product format/variant.
current_format
Include reviews relating specifically to the current format/variant.
        show_different_asins: 
Sometimes Amazon will return Reviews from ASINs other than the ASIN supplied in the asin request parameter (for example, when the original ASIN is out of stock). show_different_asins controls whether you want these other-ASIN review results returned, or not. Can be set to true (the default) to include reviews from other ASINs, or false to hide reviews from ASINs other than the ASIN supplied in the asin parameter. Note that if you supply a url instead of asin in your request this parameter is ignored.

Note in the event of `show_different_asins=false` and the ASIN shown by Amazon being different from the requested ASIN, then all fields apart from product will be removed from the response.
        review_id: A single Review ID, as returned by a prior type=reviews request, to retrieve. Useful if you wish to check for the presence of a single review. Use in combination with the amazon_domain parameter.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if global_reviews:
        querystring['global_reviews'] = global_reviews
    if review_stars:
        querystring['review_stars'] = review_stars
    if sort_by:
        querystring['sort_by'] = sort_by
    if search_term:
        querystring['search_term'] = search_term
    if page:
        querystring['page'] = page
    if reviewer_type:
        querystring['reviewer_type'] = reviewer_type
    if review_media_type:
        querystring['review_media_type'] = review_media_type
    if customer_location:
        querystring['customer_location'] = customer_location
    if gtin:
        querystring['gtin'] = gtin
    if asin:
        querystring['asin'] = asin
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if review_format:
        querystring['review_format'] = review_format
    if show_different_asins:
        querystring['show_different_asins'] = show_different_asins
    if review_id:
        querystring['review_id'] = review_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviewer_comments(asin: str, review_id: str, amazon_domain: str, type: str, sort_by: str=None, customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Review Comments Parameters are applicable when making a request with type=review_comments to retrieve customer Review Comments for a single customer review on Amazon - the review is specified using the review_id, asin and amazon_domain parameters. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Review comments are retrieved from the customer reviews page for a single customer review for a single product on Amazon."
    asin: The Amazon ASIN to retrieve Review Comments for. Used in combination with the amazon_domain and review_id parameters.
        review_id: The Review ID, as returned by a type=reviews request, to retrieve Review Comments for. Used in combination with the amazon_domain and asin parameters.
        amazon_domain: The Amazon domain to retrieve Review Comments for the Review ID specified in the review_id parameter and ASIN specified in the asin parameter. For a full list of supported Amazon domains see Supported Amazon Domains. Used in combination with the asin and review_id parameters.
        sort_by: An optional sort order for the returned review comments. Valid values are:

- most_recent 
- oldest
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \\\"France\\\" for requests for pages from \\\"amazon.fr\\\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'asin': asin, 'review_id': review_id, 'amazon_domain': amazon_domain, 'type': type, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wishlist(type: str, url: str=None, amazon_domain: str='amazon.com', wishlist_id: str='38B3V3AT7UH9B', sort_by: str=None, next_page_token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Wishlist Parameters are applicable when making a request with the type parameter set to type=wishlist Rainforest API will return data from the Wishlist page for the wishlist specified in either the wishlist_id and amazon_domain parameters or the url parameter. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Wishlist data is retrieved from the Wishlist popup."
    url: 
The Amazon wishlist-page URL to retrieve Wishlist results from.

Note: If the url parameter is supplied then the amazon_domain and wishlist_url parameters are ignored.
        amazon_domain: 
The Amazon domain to retrieve Wishlist for the product specified in the asin parameter from. For a full list of supported Amazon domains see Supported Amazon Domains.

Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.
        wishlist_id: 
The Amazon wishlist ID to retrieve Wishlist results for. Used in combination with the amazon_domain parameter. The wishlist ID can be found in any Amazon wishlist URL after the /ls/ component of the wishlist URL.

Note: If the wishlist_id and amazon_domain parameters are supplied then the url parameter is ignored.
        sort_by: 
Determines the sort order of wishlist items to return. Omit the sort_by parameter to return results in the default sort order. Valid values are:
price_high_to_low
Returns highest priced wishlist items first.
price_low_to_high
Returns lowest priced wishlist items first.
priority_high_to_low
Returns highest priority wishlist items first.
        next_page_token: the current page of Wishlist results to retrieve. Inspect the pagination.next_page_token property in the Wishlist results to retrieve the next_page_token of the subsequent page.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if wishlist_id:
        querystring['wishlist_id'] = wishlist_id
    if sort_by:
        querystring['sort_by'] = sort_by
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts(type: str, url: str='https://www.amazon.com/charts', customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Charts parameters are applicable when making a request with type=charts to retrieve Charts results. The Charts page can be specified using the url parameter. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Charts results are retrieved from the charts page on Amazon."
    url: The Charts page URL to retrieve Charts results from. Be sure to URL-encode the url parameter.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \"France\" for requests for pages from \"amazon.fr\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store(type: str, page: str=None, amazon_domain: str='amazon.com', store_id: str='70F3122D-4966-4242-A8A6-871C8D7E6F8B', url: str=None, customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Store parameters are applicable when making a request with type=store to retrieve Brand Store results for a Brand Store. The Brand Store can be specified either by the url parameter, or by a combination of the store_id and amazon_domain parameters. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request."
    page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        amazon_domain: 
The Amazon domain to retrieve Brand Store results from. For a full list of supported Amazon domains see Supported Amazon Domains. Should be used in combination with the store_id parameter.

Note if the url parameter is supplied then the amazon_domain parameter is ignored as the url itself specifies the Amazon domain.
        store_id: 
A Brand Store ID to retrieve results from. Should be used in combination with the amazon_domain parameter.

Note if the url parameter is supplied then the store_id parameter is ignored as the url itself specifies the store ID.
        url: 
The Brand Store page URL to retrieve Brand Store results from. Be sure to URL-encode the url parameter.

Note the url parameter is supplied, the store_id and amazon_domain parameters cannot be used (as the url itself defines the Store ID and Amazon domain used).
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \"France\" for requests for pages from \"amazon.fr\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if page:
        querystring['page'] = page
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if store_id:
        querystring['store_id'] = store_id
    if url:
        querystring['url'] = url
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_products(type: str, url: str=None, amazon_domain: str='amazon.com', seller_id: str='A02211013Q5HP3OMSZC7W', refinements: str=None, page: int=None, customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Seller Products Parameters are applicable when making a request with type=seller_products to retrieve seller product listing results for a single seller on Amazon - the seller is specified using either the seller_id and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon seller product listing page). The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Seller product listing results are retrieved from the seller product listings page for a single seller on Amazon. You can retrieve the seller_id value for a given seller from other Rainforest requests, such as type=offers requests."
    url: The Amazon seller products listing page URL to retrieve seller products results from.

        amazon_domain: 
The Amazon domain to retrieve seller product listings from for the seller specified in the seller_id parameter. For a full list of supported Amazon domains see Supported Amazon Domains.

Note: If the amazon_domain and seller_id parameters are supplied then the url parameter is ignored.
        seller_id: The Amazon Seller ID to retrieve seller product listings for. Used in combination with the amazon_domain parameter.

Note: If the seller_id and amazon_domain parameters are supplied then the url parameter is ignored. Note: Seller IDs can be retrieved from Rainforest API type=offers requests.
        refinements: 
A comma-seperated list of refinement values to filter the seller product results by. These allow you to refine your request by values such as \\\\\\\\\\\\\\\"Reviews rating 4 and over\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\" and \\\\\\\\\\\\\\\"brand\\\\\\\\\\\\\\\".

Refinement values are returned in the refinements array of each type=seller_products result (if they are displayed). Refinement values are dynamic and change by seller. If you wish to use refinements you should first issue a type=seller_products request without specifying any refinements to retrieve a master list of the avaialble refinements for the given request. You can then cache these refinement values for use on subsequent requests.

For example, to run a type=seller_products request specifying two refinements with values p_85/2470955011 and p_36/2421886011 the value of the refinements parameter would be refinements=p_85/2470955011,p_36/2421886011
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \"France\" for requests for pages from \"amazon.fr\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if seller_id:
        querystring['seller_id'] = seller_id
    if refinements:
        querystring['refinements'] = refinements
    if page:
        querystring['page'] = page
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_feedback(type: str, amazon_domain: str='amazon.com', page: int=None, seller_id: str='A02211013Q5HP3OMSZC7W', customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Seller Feedback Parameters are applicable when making a request with type=seller_feedback to retrieve seller customer feedback details for a seller on Amazon - the seller is specified using the seller_id and amazon_domain parameters. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Seller feedback details are retrieved from the feedback section of the seller profile page. You can retrieve the seller_id value for a given seller from other Rainforest requests, such as type=offers requests.
		
		A maximum of 5 seller feedback records can be returned per request. You can paginate through multiple pages of seller feedback records using the page parameter. Inspect the pagination.has_next_page property to determine whether there is a next page of results to retrieve."
    amazon_domain: The Amazon domain to retrieve seller feedback results from for the seller specified in the seller_id parameter. For a full list of supported Amazon domains see Supported Amazon Domains.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        seller_id: 
The Amazon Seller ID to retrieve seller feedback results for. Used in combination with the amazon_domain parameter.

Note: Seller IDs can be retrieved from Rainforest API type=offers re
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \"France\" for requests for pages from \"amazon.fr\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if page:
        querystring['page'] = page
    if seller_id:
        querystring['seller_id'] = seller_id
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_profile(type: str, url: str=None, seller_id: str='A02211013Q5HP3OMSZC7W', amazon_domain: str='amazon.com', customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Seller Profile Parameters are applicable when making a request with type=seller_profile to retrieve seller profile details for a single seller on Amazon - the seller is specified using either the seller_id and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon seller profile page). The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request.
		
		Seller profile details are retrieved from the seller profile page for a single seller on Amazon. You can retrieve the seller_id value for a given seller from other Rainforest requests, such as type=offers requests.
		
		You can use subsequent seller products and seller feedback requests to retrieve iterate all of the products and customer feedback from a given seller."
    url: The Amazon seller profile page URL of the seller to retrieve results from.

        seller_id: The Amazon Seller ID to retrieve seller profile details for. Used in combination with the amazon_domain parameter.

Note: If the seller_id and amazon_domain parameters are supplied then the url parameter is ignored. Note: Seller IDs can be retrieved from Rainforest API type=offers requests.
        amazon_domain: The Amazon domain to retrieve seller profile details from for the seller specified in the seller_id parameter. For a full list of supported Amazon domains see Supported Amazon Domains.

Note: If the amazon_domain and seller_id parameters are supplied then the url parameter is ignored.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \"France\" for requests for pages from \"amazon.fr\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if seller_id:
        querystring['seller_id'] = seller_id
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviewer_profile(type: str, reviewer_profile_pages: str=None, url: str=None, amazon_domain: str='amazon.com', reviewer_id: str='AHBEI6Q4F4WHRIFK5CWQMRVPOQMA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When making a request with the type parameter set to type=reviewer_profile Rainforest API will return details of reviewer specified in either the reviewer_id and amazon_domain parameters or the url parameter.
		
		Reviewer profile details are retrieved from the reviewer profile page. You can retrieve the reviewer_id value for a given reviewer from other Rainforest requests, such as type=reviews requests."
    reviewer_profile_pages: The number of pages of reviews to retrieve from the reviewer profile page when `type=reviewer_profile`.

This parameter is different from the more usual page parameter as pagination of reviewer profile reviews is performed from an infinate-scrolling list.

It is not possible to \\\"go to page 3\\\", so the reviewer_profile_pages allows you retrieve \\\"the first X\\\" pages of reviews (a maximum of 10 reviews are returned per page).

Note: that using reviewer_profile_pages decrements a credit for each page of reviews retrieved, for example, reviewer_profile_pages=3 would decrement 3 credits from your balance.
        url: The Amazon reviewer profile page URL of the reviewer to retrieve results from.
        amazon_domain: The Amazon domain to retrieve reviewer profile details from for the reviewer specified in the reviewer_id parameter. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and reviewer_id parameters are supplied then the url parameter is ignored.**
        reviewer_id: The Amazon Reviewer ID to retrieve reviewer profile details for. Used in combination with the amazon_domain parameter.

**Note: If the reviewer_id and amazon_domain parameters are supplied then the url parameter is ignored. Reviewer IDs can be retrieved from other Rainforest API requests, for example `type=reviews requests`.**
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if reviewer_profile_pages:
        querystring['reviewer_profile_pages'] = reviewer_profile_pages
    if url:
        querystring['url'] = url
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if reviewer_id:
        querystring['reviewer_id'] = reviewer_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bestsellers(type: str, url: str=None, page: int=2, amazon_domain: str='amazon.com', category_id: str='bestsellers_appliances', top_free: bool=None, customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Bestsellers Parameters are applicable when making a request with type=bestsellers to retrieve Bestseller results from an Amazon bestsellers page - the bestsellers page is specified either using the category_id and amazon_domain parameters (where category_id is a Category ID returned from the Categories API, or by using the url parameter. The parameters should be appended as querystring parameters to the Product Data API GET HTTP request.
		
		Note that, if using the url parameter it should be url-encoded.
		
		Bestsellers results are retrieved from the bestsellers listing page on Amazon. Rainforest supports all types of Amazon Bestseller pages, bestsellers, new releases, movers & shakers, most wished for and gift ideas."
    url: The Amazon Bestsellesr results page URL to retrieve bestsellers results from.

**Note: The url parameter should be url-encoded.**

Note: Either the url or category_id & amazon_domain parameters can be used.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        amazon_domain: The Amazon domain to retrieve bestsellers results from. For a full list of supported Amazon domains see Supported Amazon Domains.
        category_id: A category ID to retrieve bestsellers results from. You must supply a category_id returned from the Categories API in to the category_id parameter.
        top_free: Set to true to retrieve the \\\\\\\\\\\\\\\"Top 100 Free\\\\\\\\\\\\\\\" version of the specified Bestsellers category_id or url. See screenshot below for how the Top 100 Paid / Top 100 Free bestsellers for a given category are displayed on the Amazon site.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \\\"France\\\" for requests for pages from \\\"amazon.fr\\\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if page:
        querystring['page'] = page
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if category_id:
        querystring['category_id'] = category_id
    if top_free:
        querystring['top_free'] = top_free
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category(type: str, amazon_domain: str='amazon.com', url: str=None, category_id: str='2619526011', refinements: str=None, sort_by: str='average_review', page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Category Parameters are applicable when making a request with type=category to retrieve category results for an Amazon category. Categories can be specified either via passing an Amazon category URL in to the url parameter, or by passing a Category ID in to the category_id (where the category_id is returned from the Categories API) and an Amazon domain in to the amazon_domain parameter. The parameters should be appended as querystring parameters to the Product Data APIGETHTTP request."
    amazon_domain: The Amazon domain to retrieve category results from. For a full list of supported Amazon domains see Supported Amazon Domains.
        url: The Amazon category results page URL to retrieve category results from. Be sure to URL-encode the url parameter.

Note the url parameter is supplied, the category_id parameter cannot be used (as the url itself defines the category ID used).
        category_id: 
A category ID to retrieve results from. You may supply any arbitary value in the category_id parameter however we recommend using a category ID returned from the Categories API as these are known-good category IDs from Amazon.

Rainforest will use the category_id in the following form: https://amazon_domain/s?node=category_id.

Note that pagination for top-level categories is not supported by the Amazon sites. If you wish to iterate the contents of a category the recommended approach is to pick the lowest level categories to perform your iteration / pagination on.
        refinements: A comma-seperated list of refinement values to filter the category results by. These allow you to refine your category results by values such as \\\\\\\\\\\\\\\"Reviews rating 4 and over\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\" and \\\\\\\\\\\\\\\"brand\\\\\\\\\\\\\\\".

Refinement values are returned in the refinements array of each type=category result. Refinement values are dynamic and change by category area or search term used. If you wish to use refinements you should first issue a type=category request without specifying any refinements to retrieve a master list of the avaialble refinements for the given category area/search term. You can then cache these refinement values for use on subsequent requests.

For example, to run a type=search request specifying two refinements with values p_85/2470955011 and p_36/2421886011 the value of the refinements parameter would be refinements=p_85/2470955011,p_36/2421886011

Note that sometimes Amazon do not present a explicit refinement value and instead a link is returned. In this instance you should pass the link into the url request parameter of your type=category request to retrieve data from that refinement-filtered page.

        sort_by: Determines the sort order of category results to return. Valid values are:
most_recent
Sort category results by newest arrivals.
price_low_to_high
Sort category results by lowest to highest price.
price_high_to_low
Sort category results by highest to lowest price.
featured
Sort category results by featured first.
average_review
Sort category results by average customer review.
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if url:
        querystring['url'] = url
    if category_id:
        querystring['category_id'] = category_id
    if refinements:
        querystring['refinements'] = refinements
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(type: str, sort_by: str='featured', refinements: str=None, url: str=None, search_term: str='hiking backpack', amazon_domain: str='amazon.com', page: int=None, category_id: str=None, customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Search Parameters are applicable when making a request with type=search to retrieve search results for an Amazon domain - the Amazon domain is specified using the amazon_domain parameter and the search term is specified in the search_term parameter. The parameters should be appended as querystring parameters to the Product Data API GET HTTP request."
    sort_by: Determines the sort order of search results to return. Valid values are:

- most_recent
- price_low_to_high
- price_high_to_low
- featured
- average_review
- 

relevance
- Sort search results by most relevant first.
- Applies only to Audible Amazon Domains.

bestsellers
- Sort search results by best sellers first.
- Applies only to Audible Amazon Domains.

running_time_ascending
- Sort search results by shortest running time first, from short to long.
- Applies only to Audible Amazon Domains.

running_time_descending
- Sort search results by shortest running time first, from long to short.
- Applies only to Audible Amazon Domains.

title_ascending
- Sort search results alphabetically by title from A to Z.
- Applies only to Audible Amazon Domains.

title_descending
- Sort search results alphabetically by title from Z to A.
- Applies only to Audible Amazon Domains.
        refinements: A comma-seperated list of refinement values to filter the search results by. These allow you to refine your search by values such as \"Reviews rating 4 and over\", \"price range\" and \"brand\".

Refinement values are returned in the refinements array of each type=search result. Refinement values are dynamic and change by category area or search term used. If you wish to use refinements you should first issue a `type=search` request without specifying any refinements to retrieve a master list of the avaialble refinements for the given category area/search term. You can then cache these refinement values for use on subsequent requests.

For example, to run a `type=search` request specifying two refinements with values `p_85/2470955011 `and `p_36/2421886011` the value of the refinements parameter would be `refinements=p_85/2470955011,p_36/2421886011`
        url: The Amazon search results page URL to retrieve search results from. Be sure to URL-encode the url parameter.

**Note the url parameter is supplied, the search_term parameter cannot be used (as the url itself defines the search term used).**
        search_term: A search term to use to search products.
        amazon_domain: The Amazon domain to retrieve search results from. For a full list of supported Amazon domains see [Supported Amazon Domains](https://www.rainforestapi.com/docs/product-data-api/reference/amazon-domains).

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        page: The current page of search results to retrieve. Inspect the pagination.total_pages property in the Search results to see how many pages of search results are available.

update value of num to get additional pages
        category_id: A category ID to limit search results to. You may supply any arbitary value in the category_id parameter however we recommend using a category ID returned from the Categories API as these are known-good category IDs from Amazon.

Rainforest will use the category_id in the following form: `https://amazon_domain/s?node=category_id.`

Note that pagination for top-level categories is not supported by the Amazon sites. If you wish to iterate the contents of a category the recommended approach is to pick the lowest level categories to perform your iteration / pagination on.
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. \\\"France\\\" for requests for pages from \\\"amazon.fr\\\".

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if refinements:
        querystring['refinements'] = refinements
    if url:
        querystring['url'] = url
    if search_term:
        querystring['search_term'] = search_term
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if page:
        querystring['page'] = page
    if category_id:
        querystring['category_id'] = category_id
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_product(type: str, url: str=None, amazon_domain: str='amazon.com', gtin: str=None, asin: str='B073JYC4XM', customer_location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Product Parameters are applicable when making a request with type=product to retrieve details of a single product on Amazon - the product is specified using either the asin and amazon_domain parameters or the url parameter (where the url parameter contains a link to an Amazon product page). The parameters should be appended as querystring parameters to the Product Data API GET HTTP request"
    url: The Amazon product-page URL to retrieve product details from.

**Ensure that the url parameter is URL-encoded.**

Note: If the url parameter is supplied then the amazon_domain and asin parameters are ignored.
        amazon_domain: The Amazon domain to retrieve product details from for the product specified in the asin parameter. For a full list of supported Amazon domains see Supported Amazon Domains.

**Note: If the amazon_domain and asin parameters are supplied then the url parameter is ignored.**
        gtin: A GTIN, ISBN, UPC or EAN code to retrieve results for. Internally Rainforest will first convert the GTIN/ISBN/UPC/EAN to an Amazon ASIN, then retrieve the results for that ASIN from Amazon. Used in combination with the amazon_domain parameter.

**Note: If the gtin and amazon_domain parameters are supplied then the url parameter is ignored.**
        asin: The Amazon ASIN (product ID) to retrieve product details for. Used in combination with the amazon_domain parameter.

**Note: If the asin and amazon_domain parameters are supplied then the url parameter is ignored.**
        customer_location: Rainforest API supports retrieving data from the following Customer Locations.

The Customer Locations is specified via the customer_location request parameter.

The Customer Location determines the location that Rainforest uses when retrieving pages from Amazon. This is useful, for example, for seeing details of how a product appears on amazon.com, to a customer located in a different country. Can be used to identify cross-border shipping data and opportunities.

If no value for customer_location is supplied then Rainforest defaults to making the request from the country of the Amazon page requested - i.e. `France` for requests for pages from `amazon.fr`

The table below shows which customer_location values can be used in combination with requests to which amazon_domains. If you require another Amazon domain / customer location combination please contact our support team.

See the Request Parameters documentation for more information.
        
    """
    url = f"https://rainforest2.p.rapidapi.com/request"
    querystring = {'type': type, }
    if url:
        querystring['url'] = url
    if amazon_domain:
        querystring['amazon_domain'] = amazon_domain
    if gtin:
        querystring['gtin'] = gtin
    if asin:
        querystring['asin'] = asin
    if customer_location:
        querystring['customer_location'] = customer_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rainforest2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

