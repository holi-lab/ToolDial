import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getlistingoffering(product_id: int, product_offering_id: int, listing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an Offering for a Listing"
    product_id: integer >= 1
        product_offering_id: integer >= 1
        listing_id: integer >= 1
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/listings/{listing_id}/products/{product_id}/offerings/{product_offering_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getshopsection(shop_section_id: int, shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a shop section, referenced by section ID and shop ID."
    shop_section_id: integer >= 1
The numeric ID of a section in a specific Etsy shop.
        shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/sections/{shop_section_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getshopsections(shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the list of shop sections in a specific shop identified by shop ID."
    shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/sections"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findshops(shop_name: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searching shops by name. Note: We make every effort to ensure that frozen or removed shops are not included in the search results. However, rarely, due to timing issues, they may appear."
    shop_name: The shop's name string.
        offset: integer >= 0
Default: 0
The number of records to skip before selecting the first result.
        limit: integer [ 1 .. 100 ]
Default: 25
The maximum number of results to return.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops"
    querystring = {'shop_name': shop_name, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getshopbyowneruserid(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the shop identified by the shop owner's user ID."
    user_id: integer >= 1
The numeric user ID of the user who owns this shop.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/users/{user_id}/shops"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getshop(shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the shop identified by a specific shop ID."
    shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getshippingcarriers(origin_country_iso: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of available shipping carriers and the mail classes associated with them for a given country"
    origin_country_iso: string <ISO 3166-1 alpha-2>
The ISO code of the country from which the listing ships.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shipping-carriers"
    querystring = {'origin_country_iso': origin_country_iso, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreviewsbyshop(shop_id: int, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Open API to retrieve the reviews from a shop given its ID."
    shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        limit: integer [ 1 .. 100 ]
Default: 25
The maximum number of results to return.
        offset: integer >= 0
Default: 0
The number of records to skip before selecting the first result.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/reviews"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingvariationimages(listing_id: int, shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all variation images on a listing."
    listing_id: integer >= 1
The numeric ID for the listing associated to this transaction.
        shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/listings/{listing_id}/variation-images"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingtranslation(language: str, listing_id: int, shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Translation for a Listing in the given language"
    language: string
The IETF language tag for the language of this translation. Ex: de, en, es, fr, it, ja, nl, pl, pt, ru.
        listing_id: integer >= 1
The numeric ID for the listing associated to this transaction.
        shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/listings/{listing_id}/translations/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingimages(shop_id: int, listing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all listing image resources for a listing with a specific listing ID."
    shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        listing_id: integer >= 1
The numeric ID for the listing associated to this transaction.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/listings/{listing_id}/images"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingimage(shop_id: int, listing_image_id: int, listing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the references and metadata for a listing image with a specific image ID."
    shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        listing_image_id: integer >= 1
The numeric ID of the primary listing image for this transaction.
        listing_id: integer >= 1
The numeric ID for the listing associated to this transaction.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/listings/{listing_id}/images/{listing_image_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingsbyshopsectionid(shop_section_ids: str, shop_id: int, sort_order: str=None, limit: int=None, offset: int=None, sort_on: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all the listings from the section of a specific shop."
    shop_section_ids: A list of numeric IDS for all sections in a specific Etsy shop.
        shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        sort_order: Default: "desc"
Enum: "asc" "ascending" "desc" "descending" "up" "down"
The ascending(up) or descending(down) order to sort listings by. NOTE: sort_order only works when combined with one of the search options (keywords, region, etc.).
        limit: integer [ 1 .. 100 ]
Default: 25
The maximum number of results to return.
        offset: integer >= 0
Default: 0
The number of records to skip before selecting the first result.
        sort_on: Default: "created"
Enum: "created" "price" "updated"
The value to sort a search result of listings on. NOTE: sort_on only works when combined with one of the search options (keywords, region, etc.).
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/shop-sections/listings"
    querystring = {'shop_section_ids': shop_section_ids, }
    if sort_order:
        querystring['sort_order'] = sort_order
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if sort_on:
        querystring['sort_on'] = sort_on
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingproperties(listing_id: int, shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing's properties"
    listing_id: integer >= 1
The numeric ID for the listing associated to this transaction.
        shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/listings/{listing_id}/properties"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingproperty(property_id: int, listing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a listing's property"
    property_id: integer >= 1
The unique ID of an Etsy listing property.
        listing_id: integer >= 1
The numeric ID for the listing associated to this transaction.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/listings/{listing_id}/properties/{property_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfeaturedlistingsbyshop(shop_id: int, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves Listings associated to a Shop that are featured."
    shop_id: integer >= 1
The unique positive non-zero numeric ID for an Etsy Shop.
        limit: integer [ 1 .. 100 ]
Default: 25
The maximum number of results to return.
        offset: integer >= 0
Default: 0
The number of records to skip before selecting the first result.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/listings/featured"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistingsbylistingids(listing_ids: str, includes: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to query multiple listing ids at once. Limit 100 ids maximum per query."
    listing_ids: The list of numeric IDS for the listings in a specific Etsy shop.
        includes: Default: ""
Items Enum: "Shipping" "Images" "Shop" "User"
An enumerated string that attaches a valid association. Acceptable inputs are 'shop', 'images' and 'user'. Default value is an empty array.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/listings/batch"
    querystring = {'listing_ids': listing_ids, }
    if includes:
        querystring['includes'] = includes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findallactivelistingsbyshop(shop_id: str, limit: int=None, offset: int=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of all active listings on Etsy in a specific shop, paginated by listing creation date."
    shop_id: The unique positive non-zero numeric ID for an Etsy Shop.
        limit: Default: 25
The maximum number of results to return.
        offset: integer >= 0
Default: 0
The number of records to skip before selecting the first result.
        keywords: Default: null
Search term or phrase that must appear in all results.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/shops/{shop_id}/listings/active"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findalllistingsactive(sort_on: str=None, keywords: str=None, sort_order: str=None, shop_location: str=None, limit: int=None, min_price: int=None, max_price: int=None, offset: int=None, taxonomy_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all active listings on Etsy paginated by their creation date. Without sort_order listings will be returned newest-first by default."
    sort_on: Default: \\\"created\\\"
Enum: \\\"created\\\" \\\"price\\\" \\\"updated\\\"
The value to sort a search result of listings on. NOTE: sort_on only works when combined with one of the search options (keywords, region, etc.).
        keywords: Default: null
Search term or phrase that must appear in all results.
        sort_order: Default: \\\"desc\\\"
Enum: \\\"asc\\\" \\\"ascending\\\" \\\"desc\\\" \\\"descending\\\" \\\"up\\\" \\\"down\\\"
The ascending(up) or descending(down) order to sort listings by. NOTE: sort_order only works when combined with one of the search options (keywords, region, etc.).
        shop_location: Default: null
Filters by shop location. If location cannot be parsed, Etsy responds with an error.
        limit: integer [ 1 .. 100 ]
Default: 25
The maximum number of results to return.
        min_price: Default: null
The minimum price of listings to be returned by a search result.
        max_price: Default: null
The maximum price of listings to be returned by a search result.
        offset: integer >= 0
Default: 0
The number of records to skip before selecting the first result.
        taxonomy_id: Default: null
The numeric taxonomy ID of the listing. The seller manages listing taxonomy IDs for their shop. See SellerTaxonomy for more information.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/listings/active"
    querystring = {}
    if sort_on:
        querystring['sort_on'] = sort_on
    if keywords:
        querystring['keywords'] = keywords
    if sort_order:
        querystring['sort_order'] = sort_order
    if shop_location:
        querystring['shop_location'] = shop_location
    if limit:
        querystring['limit'] = limit
    if min_price:
        querystring['min_price'] = min_price
    if max_price:
        querystring['max_price'] = max_price
    if offset:
        querystring['offset'] = offset
    if taxonomy_id:
        querystring['taxonomy_id'] = taxonomy_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlisting(listing_id: int, includes: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a listing record by listing ID."
    listing_id: The numeric ID for the listing associated to this transaction.
        includes: Default: \"\"
Items Enum: \"Shipping\" \"Images\" \"Shop\" \"User\"
An enumerated string that attaches a valid association. Acceptable inputs are 'shop', 'images' and 'user'. Default value is an empty array.
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/listings/{listing_id}"
    querystring = {}
    if includes:
        querystring['includes'] = includes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpropertiesbytaxonomyid(taxonomy_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of product properties, with applicable scales and values, supported for a specific seller taxonomy ID."
    taxonomy_id: The unique numeric ID of an Etsy taxonomy node, which is a metadata category for listings organized into the seller taxonomy hierarchy tree. For example, the \"shoes\" taxonomy node (ID: 1429, level: 1) is higher in the hierarchy than \"girls' shoes\" (ID: 1440, level: 2). The taxonomy nodes assigned to a listing support access to specific standardized product scales and properties. For example, listings assigned the taxonomy nodes \"shoes\" or \"girls' shoes\" support access to the \"EU\" shoe size scale with its associated property names and IDs for EU shoe sizes, such as property value_id:\"1394\", and name:\"38\".
        
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/seller-taxonomy/nodes/{taxonomy_id}/properties"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsellertaxonomynodes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the full hierarchy tree of seller taxonomy nodes"
    
    """
    url = f"https://etsy4.p.rapidapi.com/v3/application/seller-taxonomy/nodes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

