import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def companies(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Companies monitored by DayWatch."
    format: Response format. Supported: JSON, XML, YAML
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/company/"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usa_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='women haircut', merchant_name_icontains: str='samsung', city_icontains: str='New York', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in the United States of America"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_usa/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of categories for deals."
    format: Response format. Supported: JSON, XML, YAML
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/category/"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def peru_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='corte de pelo', merchant_name_icontains: str='samsung', city_icontains: str='Lima', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in Peru"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_peru/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def colombia_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='corte de pelo', merchant_name_icontains: str='samsung', city_icontains: str='Barranquilla', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in Colombia"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_colombia/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brasil_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='convites para festa', merchant_name_icontains: str='samsung', city_icontains: str='Brasilia', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in Brasil"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_brasil/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chile_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='corte de pelo', merchant_name_icontains: str='samsung', city_icontains: str='Santiago', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in Chile"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_chile/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def argentina_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='corte de pelo', merchant_name_icontains: str='samsung', city_icontains: str='Buenos Aires', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in Argentina"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_argentina/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_summary(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resources list"
    
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mexico_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='corte de pelo', merchant_name_icontains: str='samsung', city_icontains: str='Puebla', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in Mexico"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_mexico/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uruguay_deals(format: str='json', offer_icontains: str='sushi', description_icontains: str='corte de pelo', merchant_name_icontains: str='samsung', city_icontains: str='Montevideo', discount_range: str='20,60', price_range: str='100,300', price_usd_range: str='100,300', sold_count_range: str='0,50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently active daily deals in Uruguay"
    format: Response format. Supported: JSON, XML, YAML
        offer_icontains: Filters offers that contain the given string in the offer title.
        description_icontains: Filters offers that contain the given string in the offer description.
        merchant_name_icontains: Filters offers published by the given merchant.
        city_icontains: Filters offers available in a given city.
        discount_range: Filters offers whose discount value is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_range: Filters offers whose price value in national currency is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        price_usd_range: Filters offers whose price value in US dollars is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        sold_count_range: Filters offers whose sold count is in the given range. You may also replace the __range modifier by any of the following: __gt, __gte, __lt, __lte in order to filter values greater than, greater or equal, lower than, lower or equal, respectively.
        
    """
    url = f"https://tryolabs-daywatch.p.rapidapi.com/deal_uruguay/"
    querystring = {}
    if format:
        querystring['format'] = format
    if offer_icontains:
        querystring['offer__icontains'] = offer_icontains
    if description_icontains:
        querystring['description__icontains'] = description_icontains
    if merchant_name_icontains:
        querystring['merchant_name__icontains'] = merchant_name_icontains
    if city_icontains:
        querystring['city__icontains'] = city_icontains
    if discount_range:
        querystring['discount__range'] = discount_range
    if price_range:
        querystring['price__range'] = price_range
    if price_usd_range:
        querystring['price_usd__range'] = price_usd_range
    if sold_count_range:
        querystring['sold_count__range'] = sold_count_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tryolabs-daywatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

