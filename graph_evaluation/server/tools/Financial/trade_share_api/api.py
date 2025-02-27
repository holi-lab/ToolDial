import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_publishsubscribetrades_gettrade(channelaccesscode: str=None, tradeid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    channelaccesscode: Access code to the channel where this trade is published
        tradeid: Id of the trade required
        
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetTrade"
    querystring = {}
    if channelaccesscode:
        querystring['channelAccessCode'] = channelaccesscode
    if tradeid:
        querystring['tradeID'] = tradeid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_publishsubscribetrades_gettrade(channelaccesscode: str=None, tradeid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    channelaccesscode: Access code to the channel where this trade is published
        tradeid: Id of the trade required
        
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetTrade"
    querystring = {}
    if channelaccesscode:
        querystring['channelAccessCode'] = channelaccesscode
    if tradeid:
        querystring['tradeID'] = tradeid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_publishsubscribetrades_getmyactivetrades(publishertoken: str=None, publisherkey: str=None, publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    publishertoken: Publisher token
        publisherkey: Publisher key
        publisheremail: Publisher email address
        
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetMyActiveTrades"
    querystring = {}
    if publishertoken:
        querystring['publisherToken'] = publishertoken
    if publisherkey:
        querystring['publisherKey'] = publisherkey
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_publishsubscribetrades_getmyactivetrades(publishertoken: str=None, publisherkey: str=None, publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    publishertoken: Publisher token
        publisherkey: Publisher key
        publisheremail: Publisher email address
        
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetMyActiveTrades"
    querystring = {}
    if publishertoken:
        querystring['publisherToken'] = publishertoken
    if publisherkey:
        querystring['publisherKey'] = publisherkey
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_publishsubscribetrades_getactivetrades(channelaccesscode: str=None, publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    channelaccesscode: Access code that publisher might have given to the subscriber for accessing his trades
        publisheremail: Publisher's email
        
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetActiveTrades"
    querystring = {}
    if channelaccesscode:
        querystring['channelAccessCode'] = channelaccesscode
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_publishsubscribetrades_getactivetrades(channelaccesscode: str=None, publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    channelaccesscode: Access code that publisher might have given to the subscriber for accessing his trades
        publisheremail: Publisher's email
        
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetActiveTrades"
    querystring = {}
    if channelaccesscode:
        querystring['channelAccessCode'] = channelaccesscode
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_publishsubscribetrades_gettradesignals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetTradeSignals"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_publishsubscribetrades_gettradesignals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/PublishSubscribeTrades/GetTradeSignals"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_searchpublisher(recordsperpage: int=None, searchterm: str=None, pagenumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/SearchPublisher"
    querystring = {}
    if recordsperpage:
        querystring['recordsPerPage'] = recordsperpage
    if searchterm:
        querystring['searchTerm'] = searchterm
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_searchpublisher(recordsperpage: int=None, searchterm: str=None, pagenumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/SearchPublisher"
    querystring = {}
    if recordsperpage:
        querystring['recordsPerPage'] = recordsperpage
    if searchterm:
        querystring['searchTerm'] = searchterm
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getcountofallpublishersbycountry(countryname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetCountOfAllPublishersByCountry"
    querystring = {}
    if countryname:
        querystring['countryName'] = countryname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getcountofallpublishersbycountry(countryname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetCountOfAllPublishersByCountry"
    querystring = {}
    if countryname:
        querystring['countryName'] = countryname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getcountsearchpublisher(searchterm: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetCountSearchPublisher"
    querystring = {}
    if searchterm:
        querystring['searchTerm'] = searchterm
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getcountsearchpublisher(searchterm: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetCountSearchPublisher"
    querystring = {}
    if searchterm:
        querystring['searchTerm'] = searchterm
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getpublisher(publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetPublisher"
    querystring = {}
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getpublisher(publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetPublisher"
    querystring = {}
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getallpublishersbycountry(recordsperpage: int=None, countryname: str=None, pagenumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetAllPublishersByCountry"
    querystring = {}
    if recordsperpage:
        querystring['recordsPerPage'] = recordsperpage
    if countryname:
        querystring['countryName'] = countryname
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getallpublishersbycountry(recordsperpage: int=None, countryname: str=None, pagenumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetAllPublishersByCountry"
    querystring = {}
    if recordsperpage:
        querystring['recordsPerPage'] = recordsperpage
    if countryname:
        querystring['countryName'] = countryname
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getpublisherchannels(publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetPublisherChannels"
    querystring = {}
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_management_getpublisherchannels(publisheremail: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trade-share-api.p.rapidapi.com/api/Management/GetPublisherChannels"
    querystring = {}
    if publisheremail:
        querystring['publisherEmail'] = publisheremail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trade-share-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

