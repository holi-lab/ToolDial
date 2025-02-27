import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def custom_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Custom defined endpoint, more info by contacting support."
    
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/custom"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_user_basic_parameters(filter: str='baseRegion', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Basic Parameters for Region, Currency and Locale"
    
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/user_settings"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_sku_price(itemid: int, region: str=None, currency: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_sku_price"
    querystring = {'itemId': itemid, }
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_shipping_detail_4(itemid: int, ext: str='bVkg9vW8ihgAMt4XYtZhMB3rnoY6MGI8Sk1gfrl4IGWuBdZZb0gRv5vgI1r5DIn8Rj7mxVzOKbKpyHkmBItRm_k2dtJ1j_gHLTu5zNN9jXHeQploYHEajpnygmD_xKGbi9I_HzxO8TtoIpwdvl5ZfH6o_x5qCBy5D1cUo6t7LoDhx7UTHmFDiCHY0PpHokfJ', quantity: int=1, currency: str=None, region: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        ext: Obtained from one of the **Item Detail** Endpoints. Add this value for more accuracy in shipping options.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/shipping_detail_4"
    querystring = {'itemId': itemid, }
    if ext:
        querystring['ext'] = ext
    if quantity:
        querystring['quantity'] = quantity
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_detail_5(itemid: int, locale: str=None, currency: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_detail_5"
    querystring = {'itemId': itemid, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_shipping_detail_5(ext: str, itemid: int, quantity: int=1, currency: str=None, region: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    ext: Obtained from one of the **Item Detail** Endpoints. Add this value for more accuracy in shipping options.
        itemid: Aliexpress product ID.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/shipping_detail_5"
    querystring = {'ext': ext, 'itemId': itemid, }
    if quantity:
        querystring['quantity'] = quantity
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_search_3(catid: str=None, page: int=1, loc: str=None, attr: str=None, sort: str=None, switches: str=None, startprice: int=None, q: str='iphone', endprice: int=None, brandid: str=None, locale: str=None, region: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    catid: See '**caregoryList**' from '*Item Search 2 Filters*' Endpoint for this filter value. . Eg.: 200002136


        page: Aliexpress product ID.
        loc: See '**locationList**' from '*Item Search 2 Filters*' Endpoint for this filter value. Eg.: US

        attr: See '**attributeList**'  from '*Item Search 2 Filters*' Endpoint for this filter value. Multiple values are separated with semicolons Eg.: 200000480:200004386;1186:200000072
        switches: See '**switchesList**' from the response of this Endpoint and from '*Item Search 2 Filters*' Endpoint for this filter value. Eg.: 'freeShipping' and for multiple separate with coma Eg.: 'freeShipping,sale'. 
        brandid: See '**brandList**' from the response of this Endpoint and from '*Item Search 2 Filters*' Endpoint for this filter value. Eg.: '200002136' and for multiple separate with coma Eg.: '200002136,658490794'
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_search_3"
    querystring = {}
    if catid:
        querystring['catId'] = catid
    if page:
        querystring['page'] = page
    if loc:
        querystring['loc'] = loc
    if attr:
        querystring['attr'] = attr
    if sort:
        querystring['sort'] = sort
    if switches:
        querystring['switches'] = switches
    if startprice:
        querystring['startPrice'] = startprice
    if q:
        querystring['q'] = q
    if endprice:
        querystring['endPrice'] = endprice
    if brandid:
        querystring['brandId'] = brandid
    if locale:
        querystring['locale'] = locale
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_detail_4_ru(itemid: int, locale: str=None, currency: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item focused for aliexpress.ru"
    itemid: Aliexpress product ID.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_detail_4"
    querystring = {'itemId': itemid, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_search_superdeals_plus(page: int=1, region: str=None, currency: str=None, catid: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "![https://i.ibb.co/DkxWjbD/superdeals.png](https://i.ibb.co/DkxWjbD/superdeals.png)"
    page: Aliexpress product ID.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        catid: ### SuperDeals
- 101001083571 -> Recommended
- 101001384666 -> Beauty & Health
- 101001385640 -> Fashion Style
- 101001384667 -> Mother & Kids
- 101001384668 -> Men's Clothing
- 101001385641 -> Women's Clothing
- 101001384669 -> Underwear
- 101001385642 -> Shoes
- 101001385643 -> Sports & Entertainment
- 101001384670 -> Toys & Hobbies
- 101001385644 -> Office & School
- 101001384671 -> Home Living
- 101001385645 -> DIY & Tools

### PLUS
**More to Love**
- 101001128144 -> More to Love

**Top Picks**
- 101001127194 -> Bestseller
- 101001143210 -> Tech
- 101001143280 -> Lifestyles

**Brand Deals**
- 101001124723 -> Bestseller
- 101001289254 -> Tech
- 101001289255 -> Lifestyles

**Shop by Brand**
- 31578507 -> UGREEN
- 31579070 -> BASEUS
- 31623346 -> King Billion
- 31623420 -> WORKPRO
- 31578440 -> Simplee
- 31700892 -> Podofo
- 31623343 -> Pagani Design
- 31619415 -> Wosai
- 31799223 -> Onikuma
- 31722158 -> Majestic Girls
- 31579036 -> Amazfit
- 31577866 -> Rockbros
- 31578435 -> Cupshe
- 31578288 -> Naturehike
- 31577770 -> Sofirn
- 31579013 -> 70mai
- 31577973 -> ZTTO
- 31577850 -> Svbony
- 31578271 -> Kastking
- 31623399 -> Peter Khanun



Or get json here: https://www.unserialize.com/s/266b2269-d5ca-a808-da3f-0000259d3244
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_search_promotion_deals"
    querystring = {}
    if page:
        querystring['page'] = page
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    if catid:
        querystring['catId'] = catid
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_store_item_search_by_keyword(sellerid: str, q: str, sort: str=None, page: int=None, pagesize: int=None, locale: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In-store item search by keyword."
    sellerid: Aliexpress product ID.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/store_item_search_3"
    querystring = {'sellerId': sellerid, 'q': q, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_search_by_image(imgurl: str, sort: str=None, imgregion: str=None, catid: str=None, locale: str=None, region: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search products with an Image URL. Note! You must use images from the Aliexpress products at this time."
    imgurl: The image URL must be a valid URL from Aliexpress website's products.
        imgregion: Image Region is the search focus position from the given image. Eg.: 120,45,190,150

![https://i.ibb.co/QmphdGT/img-Region-jpg.png](https://i.ibb.co/QmphdGT/img-Region-jpg.png)
        catid: See '**caregoryList**' from the response of this Endpoint for this filter value. . Eg., 4

Categories Available:
0 - Apparel
3 - Luggage & Bags
4 - Shoes
88888888 - Other

        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_search_image"
    querystring = {'imgUrl': imgurl, }
    if sort:
        querystring['sort'] = sort
    if imgregion:
        querystring['imgRegion'] = imgregion
    if catid:
        querystring['catId'] = catid
    if locale:
        querystring['locale'] = locale
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_review(itemid: int, sort: str=None, page: int=1, region: str=None, filter: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_review"
    querystring = {'itemId': itemid, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if region:
        querystring['region'] = region
    if filter:
        querystring['filter'] = filter
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_shipping_detail_3(itemid: int, ext: str, region: str=None, quantity: int=1, locale: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        ext: Obtained from one of the **Item Detail** Endpoints. Add this value for more accuracy in shipping options.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/shipping_detail_3"
    querystring = {'itemId': itemid, 'ext': ext, }
    if region:
        querystring['region'] = region
    if quantity:
        querystring['quantity'] = quantity
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_shipping_detail_2(itemid: int, quantity: int=1, currency: str=None, ext: str='bVkg9vW8ihgAMt4XYtZhMB3rnoY6MGI8Sk1gfrl4IGWuBdZZb0gRv5vgI1r5DIn8Rj7mxVzOKbKpyHkmBItRm_k2dtJ1j_gHLTu5zNN9jXHeQploYHEajpnygmD_xKGbi9I_HzxO8TtoIpwdvl5ZfH6o_x5qCBy5D1cUo6t7LoDhx7UTHmFDiCHY0PpHokfJ', locale: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        ext: Obtained from one of the **Item Detail** Endpoints. Add this value for more accuracy in shipping options.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/shipping_detail_2"
    querystring = {'itemId': itemid, }
    if quantity:
        querystring['quantity'] = quantity
    if currency:
        querystring['currency'] = currency
    if ext:
        querystring['ext'] = ext
    if locale:
        querystring['locale'] = locale
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_store_item_search_2(sort: str=None, locale: str=None, storeid: str='1102051418', page: int=None, pagesize: int=None, currency: str=None, region: str=None, sellerid: str='231651707', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        sellerid: Aliexpress product ID.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/store_item_search_2"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if locale:
        querystring['locale'] = locale
    if storeid:
        querystring['storeId'] = storeid
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    if sellerid:
        querystring['sellerId'] = sellerid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_store_categories(sellerid: str, storeid: str=None, region: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    sellerid: Aliexpress product ID.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/store_categories"
    querystring = {'sellerId': sellerid, }
    if storeid:
        querystring['storeId'] = storeid
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_shipping_detail(itemid: int, quantity: int=1, sellerid: str=None, ext: str='bVkg9vW8ihgAMt4XYtZhMB3rnoY6MGI8Sk1gfrl4IGWuBdZZb0gRv5vgI1r5DIn8Rj7mxVzOKbKpyHkmBItRm_k2dtJ1j_gHLTu5zNN9jXHeQploYHEajpnygmD_xKGbi9I_HzxO8TtoIpwdvl5ZfH6o_x5qCBy5D1cUo6t7LoDhx7UTHmFDiCHY0PpHokfJ', locale: str=None, region: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        ext: Obtained from one of the **Item Detail** Endpoints. Add this value for more accuracy in shipping options.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/shipping_detail"
    querystring = {'itemId': itemid, }
    if quantity:
        querystring['quantity'] = quantity
    if sellerid:
        querystring['sellerId'] = sellerid
    if ext:
        querystring['ext'] = ext
    if locale:
        querystring['locale'] = locale
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_detail_2(itemid: int, currency: str=None, region: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item. This Endpoint needs an additional request to the "Item Description" endpoint to fill the *Properties* and *Description* gaps."
    itemid: Aliexpress product ID.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_detail_2"
    querystring = {'itemId': itemid, }
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_search(brandid: str=None, catid: str=None, q: str='iphone', page: int=1, loc: str=None, locale: str=None, endprice: int=None, sort: str=None, attr: str=None, startprice: int=None, region: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    catid: See '**caregoryList**' from response for this filter values. Eg.: 200002136


        page: Aliexpress product ID.
        loc: See '**locationList**' from the response for this filter value. Eg.: US

        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        attr: See '**attributeList**' from the response for this filter value. Multiple values are separate with semicolons Eg.: 200000480:200004386;1186:200000072
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_search"
    querystring = {}
    if brandid:
        querystring['brandId'] = brandid
    if catid:
        querystring['catId'] = catid
    if q:
        querystring['q'] = q
    if page:
        querystring['page'] = page
    if loc:
        querystring['loc'] = loc
    if locale:
        querystring['locale'] = locale
    if endprice:
        querystring['endPrice'] = endprice
    if sort:
        querystring['sort'] = sort
    if attr:
        querystring['attr'] = attr
    if startprice:
        querystring['startPrice'] = startprice
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_store_info(sellerid: str, locale: str=None, region: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    sellerid: Aliexpress product ID.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/store_info"
    querystring = {'sellerId': sellerid, }
    if locale:
        querystring['locale'] = locale
    if region:
        querystring['region'] = region
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_review_2(itemid: int, filter: str=None, locale: str=None, page: int=1, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_review_2"
    querystring = {'itemId': itemid, }
    if filter:
        querystring['filter'] = filter
    if locale:
        querystring['locale'] = locale
    if page:
        querystring['page'] = page
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_description(itemid: int, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_desc"
    querystring = {'itemId': itemid, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_detail_simple(itemid: int, locale: str=None, currency: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_detail_simple"
    querystring = {'itemId': itemid, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_search_2(endprice: int=None, sort: str=None, brandid: str=None, page: int=1, attr: str=None, startprice: int=None, locale: str=None, switches: str=None, catid: str=None, q: str='iphone', loc: str=None, currency: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    brandid: See '**brandList**' from the response of this Endpoint and from '*Item Search 2 Filters*' Endpoint for this filter value. Eg.: '200002136' and for multiple separate with coma Eg.: '200002136,658490794'
        page: Aliexpress product ID.
        attr: See '**attributeList**'  from '*Item Search 2 Filters*' Endpoint for this filter value. Multiple values are separated with semicolons Eg.: 200000480:200004386;1186:200000072
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        switches: See '**switchesList**' from the response of this Endpoint and from '*Item Search 2 Filters*' Endpoint for this filter value. Eg.: 'freeShipping' and for multiple separate with coma Eg.: 'freeShipping,sale'. 
        catid: See '**caregoryList**' from '*Item Search 2 Filters*' Endpoint for this filter value. . Eg.: 200002136


        loc: See '**locationList**' from '*Item Search 2 Filters*' Endpoint for this filter value. Eg.: US

        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_search_2"
    querystring = {}
    if endprice:
        querystring['endPrice'] = endprice
    if sort:
        querystring['sort'] = sort
    if brandid:
        querystring['brandId'] = brandid
    if page:
        querystring['page'] = page
    if attr:
        querystring['attr'] = attr
    if startprice:
        querystring['startPrice'] = startprice
    if locale:
        querystring['locale'] = locale
    if switches:
        querystring['switches'] = switches
    if catid:
        querystring['catId'] = catid
    if q:
        querystring['q'] = q
    if loc:
        querystring['loc'] = loc
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_store_item_search(sellerid: str, storeid: str, sort: str=None, page: int=None, pagesize: int=None, region: str=None, locale: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    sellerid: Aliexpress product ID.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/store_item_search"
    querystring = {'sellerId': sellerid, 'storeId': storeid, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    if region:
        querystring['region'] = region
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_detail_3(itemid: int, currency: str=None, region: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_detail_3"
    querystring = {'itemId': itemid, }
    if currency:
        querystring['currency'] = currency
    if region:
        querystring['region'] = region
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aliexpress_item_detail(itemid: int, currency: str=None, locale: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item."
    itemid: Aliexpress product ID.
        currency: Currency parameter is used to display the product price in selected currency. All values are being validated and only the ones from the list are supported. Full list of supporting currencies, please use '**Currency List**' at '**Base**' group endpoints.
        locale: Locale parameter is used to display the titles and other content in selected language. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Locale List'** at '**Base**' group endpoints.
        region: Region parameter is used to accurately get the shipping costs and other promo/deals for specific geo location. All values are being validated and only the ones from the list are supported. Full list of supporting locale options, please use '**Country List**' at '**Base**' group endpoints.
        
    """
    url = f"https://aliexpress-datahub.p.rapidapi.com/item_detail"
    querystring = {'itemId': itemid, }
    if currency:
        querystring['currency'] = currency
    if locale:
        querystring['locale'] = locale
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-datahub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

