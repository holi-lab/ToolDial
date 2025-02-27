import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_product_productid(productid: int, targetcurrency: str=None, lg: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents a product, each product is identifier by ID"
    productid: The Aliexpress product ID
        targetcurrency: Represents the currency whished when the product is retrieved
        lg: Represents the language wished when displaying the single product. He must compliant with ISO 639-1 either have two digits
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/product/{productid}"
    querystring = {}
    if targetcurrency:
        querystring['targetCurrency'] = targetcurrency
    if lg:
        querystring['lg'] = lg
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_store_storeid_seller_sellerid_products(storeid: int, page: str='1', sellerid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This ressource displays the products by Store"
    storeid: The Aliexpress store ID
        sellerid: You can found the ID in the attribute under sellerAdminSeq, this one  can be found  in the /api/product/{productId}  in the storeModule inside the metadata.

        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/store/{storeid}/seller/{sellerID}/products"
    querystring = {}
    if page:
        querystring['page'] = page
    if sellerid:
        querystring['sellerID'] = sellerid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_products_search(name: str, getshopinformation: bool=None, lg: str=None, shiptocountry: str='FR', minsaleprice: int=5, targetcurrency: str=None, shipfromcountry: str='CN', fastdelivery: bool=None, maxsaleprice: int=None, sort: str='NEWEST_DESC', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search a product by his name"
    name: Filter the products compatible with the name
        shiptocountry: Filter the products that can be to ship to the country wished. you can use the service countriesAvailable to set the country wished.
        minsaleprice: Filter products by the min Sale price
        targetcurrency: Represents the currency whished when the product is retrieved
        shipfromcountry: Filter the products that can be to ship from  country wished. you can use the service countriesAvailable to set the good information.
        fastdelivery: Filter products that are compatible with fast delivery
        maxsaleprice: Filter products by the max Sale price
        sort: the products are sorted
        page: Represents the page wished
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/products/search"
    querystring = {'name': name, }
    if getshopinformation:
        querystring['getShopInformation'] = getshopinformation
    if lg:
        querystring['lg'] = lg
    if shiptocountry:
        querystring['shipToCountry'] = shiptocountry
    if minsaleprice:
        querystring['minSalePrice'] = minsaleprice
    if targetcurrency:
        querystring['targetCurrency'] = targetcurrency
    if shipfromcountry:
        querystring['shipFromCountry'] = shipfromcountry
    if fastdelivery:
        querystring['fastDelivery'] = fastdelivery
    if maxsaleprice:
        querystring['maxSalePrice'] = maxsaleprice
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_product_productid_feedbacks(productid: int, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This ressource displays the feedbacks for one product"
    productid: The Aliexpress product ID
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/product/{productid}/feedbacks"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_shipping_productid(productid: int, tocountry: str='FR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This ressource displays the shipping information for one product"
    productid: The Aliexpress product ID
        tocountry: The country to calcul the freight 
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/shipping/{productid}"
    querystring = {}
    if tocountry:
        querystring['toCountry'] = tocountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_category_categoryid_products(categoryid: int, brand: str=None, shiptocountry: str='FR', attr: str=None, maxsaleprice: int=50, getshopinformation: bool=None, targetcurrency: str=None, lg: str=None, shipfromcountry: str='CN', minsaleprice: int=10, sort: str='NEWEST_DESC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents the product list; you can filter by
		 - attributes
		 - minimum price
		 - brand 
		 - ship to country
		 - ship from country
		 - max sale price 
		<br> **For information, One first request, you must not indicate the attr & brands, you retrieve this informations in the Product Object.**"
    categoryid: The Aliexpress product ID
        brand: filter the products by brand. <br> In order todo this :<br> 1. Filter the products with all filters that you want,  without **brand** <br> 2. Search the `brands` element at the same level that pagination level (limit, page hasPrevPage ... ) <br> 3. Indicate the content id  <br>content:[<br>{ <br>brandName:MEGE KNIGHT<br>**brandId:201549252** <br>brandLogo://ae01.alicdn.com/kf/HTB1XyEwE1OSBuNjy0Fdq6zDnVXaw.jpg<br>}<br>If you want filter by the MEGE KNIGHT you must indicate 201549252. <br> **Warning, when you indicate brand element, the attr parameter must be empty.**
        shiptocountry: Filter the products that can be to ship to the country wished. you can use the service countriesAvailable to set the good information
        attr: filter the products by an attribute.<br> In order todo this :<br>1. Filter the products with all filters that you want,  without **attr** <br>2. Search the `attributes` element at the same level that pagination level (limit, page hasPrevPage ... ) <br>3. concatenate the content parent with children element <br>content:[<br>{ <br>**attributeId:14** <br>displayType:colour_atla <br>attributeValues:[ <br>{<br>**attributeValueId:10**<br>attributeValueName:Red<br>selected:true<br>}<br>If you want filter the products by the color red, you must indicate 14-10.<br> **Warning, when you indicate attribute element, the brand parameter must be empty.**
        maxsaleprice: Filter products by the max Sale price
        targetcurrency: Represents the currency whished when the product is retrieved
        shipfromcountry: Filter the products that can be to ship from  country wished. you can use the service countriesAvailable to set the good information
        minsaleprice: Filter products by the min Sale price
        sort: filter the products
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/category/{categoryid}/products"
    querystring = {}
    if brand:
        querystring['brand'] = brand
    if shiptocountry:
        querystring['shipToCountry'] = shiptocountry
    if attr:
        querystring['attr'] = attr
    if maxsaleprice:
        querystring['maxSalePrice'] = maxsaleprice
    if getshopinformation:
        querystring['getShopInformation'] = getshopinformation
    if targetcurrency:
        querystring['targetCurrency'] = targetcurrency
    if lg:
        querystring['lg'] = lg
    if shipfromcountry:
        querystring['shipFromCountry'] = shipfromcountry
    if minsaleprice:
        querystring['minSalePrice'] = minsaleprice
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_bestsales_sortedbynewest(limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the last best product added in the system"
    
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/bestSales/SortedByNewest"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_product_productid_historic_prices(productid: str, maxdate: str=None, mindate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the prices historic of product loaded by a client"
    
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/product/{productid}/historic/prices"
    querystring = {}
    if maxdate:
        querystring['maxDate'] = maxdate
    if mindate:
        querystring['minDate'] = mindate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_product_productid_historic_sales(productid: str, mindate: str=None, maxdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the sales historic of product loaded by a client"
    
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/product/{productid}/historic/sales"
    querystring = {}
    if mindate:
        querystring['minDate'] = mindate
    if maxdate:
        querystring['maxDate'] = maxdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents a category in the system. We have 3 provider type <br> - API
		 -WEBSITE
		 -HYBRID
		 <br> 1.API<br> This category is manager by the API, the ID indicated is the Alibaba ID and not Aliexpress ID.<br> 2. WESITE <br> This category is scraping of Aliexpress website <br> 3. HYBRID <br> This category has correspondence between Aliexpress and API; you have in the same object twice ID `alie_category_id` for Website and `api_category_id` for API<br> In the API to request by example product, in the category ID, you can indicate the both.  "
    
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/v2/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_currenciesavailable(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents a currency in the system. Each currency is composed of Code and Name Attributes. This service can be used to communicate with product service with the target currency attribute."
    
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/currenciesAvailable"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_countriesavailabletoshipping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents the countries. Each country is composed of Code and Name Attributes. This service can be used to communicate with product service for the countryFrom and countryTo query parameters."
    
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/countriesAvailableToShipping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_bestsales_product_productid_saleshistory(productid: int, mindate: str='2020-09-28', maxdate: str='2021-08-05', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource returns the historic of sales"
    productid: The Aliexpress product ID
        mindate: filter the historic by min date
        maxdate: filter the historic by max date
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/bestSales/product/{productid}/salesHistory"
    querystring = {}
    if mindate:
        querystring['minDate'] = mindate
    if maxdate:
        querystring['maxDate'] = maxdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_bestsales_products(page: int, pricemin: int=5, categoryid: int=None, pricemax: int=20, sort: str='EVALUATE_RATE_ASC', searchname: str='shoes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents the best sales detected by our system"
    page: you can sorted the products
        pricemin: filter by the price min
        categoryid: filter by the category's ID
        pricemax: filter by the price max
        sort: you can sorted the products
        searchname: Returns products with the name
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/bestSales/products"
    querystring = {'page': page, }
    if pricemin:
        querystring['priceMin'] = pricemin
    if categoryid:
        querystring['categoryID'] = categoryid
    if pricemax:
        querystring['priceMax'] = pricemax
    if sort:
        querystring['sort'] = sort
    if searchname:
        querystring['searchName'] = searchname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_bestsales_product_productid_priceshistory(productid: int, mindate: str='2020-09-28', maxdate: str='2021-08-05', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource returns the historic of prices"
    productid: The Aliexpress product ID
        mindate: filter the historic min date
        maxdate: filter the historic by max date
        
    """
    url = f"https://magic-aliexpress1.p.rapidapi.com/api/bestSales/product/{productid}/pricesHistory"
    querystring = {}
    if mindate:
        querystring['minDate'] = mindate
    if maxdate:
        querystring['maxDate'] = maxdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-aliexpress1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

