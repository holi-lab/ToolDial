import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_amazon_product_details(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A product detail page is where a customer discovers a unique product sold on Amazon. It can include one or more offers from sellers or from Amazon.
		
		This page is a shared space that displays attributes that are common to all offers for the product, such as the following:
		
		Title
		Image
		Bullet points
		Description
		Product variations (such as size or colour)
		Customer reviews
		Some categories have additional product detail attributes. For example in Electronics, cameras have attributes such as optical zoom and max resolution that would not be required for other product categories.
		
		You and other sellers can list an offer on a product detail page. You create and control your own offer for a product, including price, shipping options, condition and other attributes. If a product does not exist on Amazon, you can create an offer for it and Amazon will create a new product detail page. Amazon chooses what information to include on the product detail page based on manufacturer and seller contributions."
    
    """
    url = f"https://amazon-data-product-scraper.p.rapidapi.com/products/{productid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data-product-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_reviews(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Customer Reviews help customers to learn more about the product and decide whether it is the right product for them.
		
		Customer Reviews should give customers genuine product feedback from fellow shoppers. We have a zero tolerance policy for any review designed to mislead or manipulate customers.
		
		We don't allow anyone to write reviews as a form of promotion.
		
		The following are types of reviews that we don't allow and will remove:
		
		A review by someone who has a direct or indirect financial interest in the product.
		A review by someone perceived to have a close personal relationship with the product's owner, author, or artist.
		A review by the product manufacturer, posing as an unbiased shopper.
		Multiple negative reviews for the same product from one customer.
		A review in exchange for monetary reward.
		A review of a game in exchange for bonus in-game credits.
		A negative review from a seller on a competitor's product.
		A positive review from an artist on a peer's album in exchange for receiving a positive review from them."
    
    """
    url = f"https://amazon-data-product-scraper.p.rapidapi.com/products/{productid}/reviews"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data-product-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_offers(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A product detail page is where a customer discovers a unique product sold on Amazon. It can include one or more offers from sellers or from Amazon.
		
		This page is a shared space that displays attributes that are common to all offers for the product, such as the following:
		
		Title
		Image
		Bullet points
		Description
		Product variations (such as size or colour)
		Customer reviews
		Some categories have additional product detail attributes. For example in Electronics, cameras have attributes such as optical zoom and max resolution that would not be required for other product categories.
		
		You and other sellers can list an offer on a product detail page. You create and control your own offer for a product, including price, shipping options, condition and other attributes. If a product does not exist on Amazon, you can create an offer for it and Amazon will create a new product detail page. Amazon chooses what information to include on the product detail page based on manufacturer and seller contributions."
    
    """
    url = f"https://amazon-data-product-scraper.p.rapidapi.com/products/{productid}/offers"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data-product-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_search_results(searchquery: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In order for customers to find your products on Amazon, it’s important to provide Search Terms they might use when searching for what they want to buy.
		
		Amazon limits the length of the Search Terms attribute to less than 250 bytes. The Search Terms limit applies to newly-registered as well as existing ASINs.
		
		Tips for optimizing your Search Terms:
		
		Stay under length limit.
		Include synonyms.
		Include spelling variations, no need for misspellings.
		Include abbreviations and alternate names.
		You can use all lower case.
		You don’t need punctuation, such as: ";", ":", "-".
		Separate words with spaces.
		Don’t repeat words within the Search Terms field.
		No need for stop words such as “a,” “an,” “and,” “by,” “for,” “of,” “the,” “with,” and so on.
		Use singular or plural, no need for both.
		Prohibited use of Search Terms (keywords):
		
		Violating these rules may result in your ASIN being suppressed and action against your account.
		
		Don’t include your brand or other brand names in Search Terms.
		Don’t include ASINs in Search Terms.
		Don’t add profanity
		No temporary statements such as “new,” or “on sale now.”
		Don’t use subjective claims, such as “best,” “cheapest,” “amazing,” and so on.
		Don’t add abusive or offensive terms.
		Brand Names:	Apple, Nike, Amazon, etc.
		Temporary Words:	available now, brand new, current, discounted, just launched, last chance, last minute, latest, limited time, new, on sale, this week (month, year etc.), today, etc.
		Subjective Words:	amazing, best, cheap, cheapest, effective, fastest, good deal, least, most, popular, trending, etc."
    
    """
    url = f"https://amazon-data-product-scraper.p.rapidapi.com/search/{searchquery}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data-product-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

