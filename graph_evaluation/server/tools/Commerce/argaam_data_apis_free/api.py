import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_list_of_bank_sector_groups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Bank Sector Groups"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/facts-statistics/get-bank-sector-reports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_banks_sector_attributes_against_each_group(reportid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Banks Sector Attributes against each group"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/facts-statistics/get-bank-sector-report/{reportid}/attributes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_indicator_groups_and_subgroups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Indicator Groups and Subgroups"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/indicators/get-groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_indicator_against_each_subgroup(groupid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Indicator against each Subgroup"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/indicators/get-group-fields/{groupid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_fund_managers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Fund Managers"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/funds/get-fund-managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_funds_against_each_fund_manager(fundmanagerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Funds against each Fund Manager"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/funds/get-fund-manager/{fundmanagerid}/funds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_cement_attribute_groups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Cement Attribute Groups"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/facts-statistics/get-cement-companies-reports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_cement_attributes_against_each_group(reportid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Cement Attributes against each group"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/facts-statistics/get-cement-company-report/{reportid}/attributes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_bank_attributes_groups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Bank Attributes Groups"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/facts-statistics/get-bank-companies-reports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_banks_attributes_against_each_group(reportid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a List of Banks Attributes against each group"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/facts-statistics/get-bank-company-report/{reportid}/attributes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_popular_articles_by_market(lang: str='ar', marketid: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Popular Articles by Market"
    lang: Articles Required in Language. Options: 'en' and 'ar'. Default Value: 'ar'
        marketid: Market ID for Popular Articles
        
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/articles/get-popular-articles"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if marketid:
        querystring['marketID'] = marketid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_s_profile(marketid: int, companyid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Company's Profile"
    marketid: Company's Market ID
        companyid: Company ID
        
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/company/get-profile/{marketid}/{companyid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_consumer_goods_with_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Consumer Goods with Categories"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/goods/get-consumer-goods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_markets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Markets"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/markets/get-markets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_commodities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Commodities"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/commodities/get-commodities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_sources(page: int=1, lang: str='ar', records: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Article Sources"
    page: Page No. Default: 1
        lang: Language in which Article Sources are to be fetched. Default: 'ar'. Options: 'ar', 'en'
        records: Records per page. Default: 200
        
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/articles/get-article-sources"
    querystring = {}
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    if records:
        querystring['records'] = records
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_international_market_data(marketid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get International Market Data"
    marketid: International Market ID for which Data is needed.
        
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/markets/get-international-markets-data/{marketid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_market_data(marketid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Market Data"
    marketid: Market ID for which Data is needed.
        
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/markets/get-markets-data/{marketid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Article Types"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/articles/get-article-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_inernational_markets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get International Markets"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/markets/get-international-markets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_petrochemical_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Petrochemical Index"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/petrochemicals/index"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sectors_by_market(marketid: int, sectorid: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Sectors by Market"
    marketid: Market ID for which Sectors need to fetch.
        sectorid: Optional SectorID to get specific sector details
        
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/sectors/get-sectors-by-market/{marketid}"
    querystring = {}
    if sectorid:
        querystring['sectorID'] = sectorid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sector_s_data(marketid: int, sectorid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Sector's Data"
    marketid: Market ID for which Sectors Data need to fetch.
        sectorid: Optional SectorID to get specific sector details
        
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/sectors/get-sector-data/{marketid}/{sectorid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_regular_commodities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get List of Regular Commodities with Quotes"
    
    """
    url = f"https://argaam-data-apis-free.p.rapidapi.com/api/v1/json/commodities/get-regular-commodity-quotes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argaam-data-apis-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

