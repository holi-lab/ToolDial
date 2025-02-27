import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_lists(ws_op: str, accountid: int, ws_size: int=None, name: str=None, ws_show: str=None, ws_start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a list within an account"
    ws_op: Search
        accountid: Account ID
        ws_size: Number of items to retreive from pagination
        name: Name or unique list id
        ws_show: Return the total size of the collection only
        ws_start: Starting position for pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists"
    querystring = {'ws.op': ws_op, }
    if ws_size:
        querystring['ws.size'] = ws_size
    if name:
        querystring['name'] = name
    if ws_show:
        querystring['ws.show'] = ws_show
    if ws_start:
        querystring['ws.start'] = ws_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_links(listid: int, campaigntype: str, accountid: int, campaignid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the links for a broadcast or follow up within a list"
    listid: List ID
        campaigntype: Campaign Type
        accountid: Account ID
        campaignid: Campaign ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/links"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_link(campaigntype: str, listid: int, campaignid: int, linkid: int, accountid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific link from a broadcast or follow up within a list"
    campaigntype: Campaing Type
        listid: List ID
        campaignid: Campaign ID
        linkid: Link ID
        accountid: Account ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/links/{linkid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_message_tracked_events(accountid: int, campaignid: int, listid: int, campaigntype: str, messageentryid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the tracked events associated with a message for a broadcast or follow up within a list"
    accountid: Account ID
        campaignid: Campaign ID
        listid: List ID
        campaigntype: Campaign Type
        messageentryid: Message Entry ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/messages/{messageentryid}/tracked_events"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_broadcast_statistic(accountid: int, statsid: str, listid: int, campaignid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific statistic for a broadcast within a list"
    accountid: Account ID
        statsid: Stats ID
        listid: List ID
        campaignid: Campaign ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/b{campaignid}/stats/{statsid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_webform_for_list(accountid: int, listid: int, webformid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific webform within a list"
    accountid: Account ID
        listid: List ID
        webformid: Web Form ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/web_forms/{webformid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_message_opens(accountid: int, listid: int, campaignid: int, campaigntype: str, messageentryid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the opens related to a specific message for a broadcast or follow up within a list"
    accountid: Account ID
        listid: List ID
        campaignid: Campaign ID
        campaigntype: Campaign Type
        messageentryid: Message Entry ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/messages/{messageentryid}/opens"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_message_tracked_event(campaignid: int, accountid: int, listid: int, trackedeventid: int, messageentryid: int, campaigntype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific tracked event associated with a message for a broadcast or follow up within a list"
    campaignid: Campaign ID
        accountid: Account ID
        listid: List ID
        trackedeventid: Tracked Event ID
        messageentryid: Message Entry ID
        campaigntype: Campaign Type
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/messages/{messageentryid}/tracked_events/{trackedeventid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_split_test_components(accountid: int, listid: int, splittestid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the components of a web form split test"
    accountid: Account ID
        listid: List ID
        splittestid: Split Test ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/web_form_split_tests/{splittestid}/components"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list(listid: int, accountid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific list for an account"
    listid: List ID
        accountid: Account ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_split_test_component(splittestid: int, listid: int, accountid: int, splittestcomponentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific component of a web form split test"
    splittestid: Split Test ID
        listid: List ID
        accountid: Account ID
        splittestcomponentid: Split Test Component ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/web_form_split_tests/{splittestid}/components/{splittestcomponentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_segments(accountid: int, listid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the segments within a list"
    accountid: Account ID
        listid: List ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/segments"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subscriber(accountid: int, subscriberid: int, listid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific subscriber within a list"
    accountid: Account ID
        subscriberid: Subscriber ID
        listid: List ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/subscribers/{subscriberid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_split_tests_for_account(ws_op: str, accountid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the web form split tests for an account"
    ws_op: Specify you want web form split tests
        accountid: Account ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}"
    querystring = {'ws.op': ws_op, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subscribers(listid: int, accountid: int, ws_start: int=None, ws_size: int=None, sort_order: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the subscribers witihin a list"
    listid: List ID
        accountid: Account ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        sort_order: Specify the sort order of the data
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/subscribers"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_integration(integrationid: int, accountid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an integration within an account"
    integrationid: Integration ID
        accountid: Account ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/integrations/{integrationid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_campaigns(ws_op: str, campaign_type: str, listid: int, accountid: int, ws_start: int=None, ws_show: str=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a follow up or broadcast within a list"
    ws_op: Specify the desire to search
        campaign_type: Specify a follow up ("f") or a broadcast ("b")
        listid: List ID
        accountid: Account ID
        ws_start: Starting position for pagination
        ws_show: Return only the total number of rows
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns"
    querystring = {'ws.op': ws_op, 'campaign_type': campaign_type, }
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_show:
        querystring['ws.show'] = ws_show
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_subscribers_for_list(ws_op: str, listid: int, accountid: int, ws_size: int=None, sort_order: str=None, ws_show: bool=None, ad_tracking: str=None, city: str=None, country: str=None, sort_key: str=None, area_code: int=None, ws_start: int=None, email: str=None, last_followup_message_sent_at: str=None, dma_code: int=None, longitude: str=None, latitude: int=None, misc_notes: str=None, name: str=None, custom_fields: str=None, status: str=None, region: str=None, subscribed_after: str=None, subscribed_at: str=None, postal_code: str=None, last_followup_message_number_sent: int=None, subscription_method: str=None, unsubscribe_method: str=None, subscribed_before: str=None, unsubscribed_after: str=None, verified_at: str=None, unsubscribed_before: str=None, unsubscribed_at: str=None, tags: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for subscribers within a list"
    ws_op: Specify that you want to search
        listid: Lists ID
        accountid: Account ID
        ws_size: Number of items to retrieve from pagination
        sort_order: The direction of the sort order
        ws_show: Return just the total number of rows
        ad_tracking: Search for an ad_tracking value
        city: Search for a city
        country: Search for a country
        sort_key: The field to be sorted
        area_code: Search for an area code
        ws_start: Starting position for pagination
        email: Search for an email address
        last_followup_message_sent_at: Search the date of the last follow-up message sent
        dma_code: Search for a dma code
        longitude: Search for a specific longitude
        latitude: Search for a specific latitude
        misc_notes: Search for miscellenous notes
        name: Search the subscriber's name
        custom_fields: Search custom field values
        status: Search the subscriber's status
        region: Search the region
        subscribed_after: Search all subscriber who subscribed after a date
        subscribed_at: Search all subscribers who subscribed on a date
        postal_code: Search the postal code
        last_followup_message_number_sent: Search for the last following message number sent
        subscription_method: Search for the subscription method
        unsubscribe_method: Search for the unsubscribe method
        subscribed_before: Search all subscribers who subscribed before a date
        unsubscribed_after: Search for subscribers who unsubscribed after a date
        verified_at: Search for subscribers who verified their email on a date
        unsubscribed_before: Search for subscribers who unsubscribed before a date
        unsubscribed_at: Search for subscribers who unsubscribed on a date
        tags: Search for a subscriber's tags
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/subscribers"
    querystring = {'ws.op': ws_op, }
    if ws_size:
        querystring['ws.size'] = ws_size
    if sort_order:
        querystring['sort_order'] = sort_order
    if ws_show:
        querystring['ws.show'] = ws_show
    if ad_tracking:
        querystring['ad_tracking'] = ad_tracking
    if city:
        querystring['city'] = city
    if country:
        querystring['country'] = country
    if sort_key:
        querystring['sort_key'] = sort_key
    if area_code:
        querystring['area_code'] = area_code
    if ws_start:
        querystring['ws.start'] = ws_start
    if email:
        querystring['email'] = email
    if last_followup_message_sent_at:
        querystring['last_followup_message_sent_at'] = last_followup_message_sent_at
    if dma_code:
        querystring['dma_code'] = dma_code
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    if misc_notes:
        querystring['misc_notes'] = misc_notes
    if name:
        querystring['name'] = name
    if custom_fields:
        querystring['custom_fields'] = custom_fields
    if status:
        querystring['status'] = status
    if region:
        querystring['region'] = region
    if subscribed_after:
        querystring['subscribed_after'] = subscribed_after
    if subscribed_at:
        querystring['subscribed_at'] = subscribed_at
    if postal_code:
        querystring['postal_code'] = postal_code
    if last_followup_message_number_sent:
        querystring['last_followup_message_number_sent'] = last_followup_message_number_sent
    if subscription_method:
        querystring['subscription_method'] = subscription_method
    if unsubscribe_method:
        querystring['unsubscribe_method'] = unsubscribe_method
    if subscribed_before:
        querystring['subscribed_before'] = subscribed_before
    if unsubscribed_after:
        querystring['unsubscribed_after'] = unsubscribed_after
    if verified_at:
        querystring['verified_at'] = verified_at
    if unsubscribed_before:
        querystring['unsubscribed_before'] = unsubscribed_before
    if unsubscribed_at:
        querystring['unsubscribed_at'] = unsubscribed_at
    if tags:
        querystring['tags'] = tags
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_integrations(accountid: int, ws_size: int=None, ws_start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the social integrations in an account"
    accountid: Account ID
        ws_size: Number of items to retrieve from pagination
        ws_start: Starting position for pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/integrations"
    querystring = {}
    if ws_size:
        querystring['ws.size'] = ws_size
    if ws_start:
        querystring['ws.start'] = ws_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_click(linkid: int, listid: int, accountid: int, campaignid: int, campaigntype: str, clickid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specifc click for a broadcast or follow up within a list"
    linkid: Link ID
        listid: List ID
        accountid: Account ID
        campaignid: Campaign ID
        campaigntype: Campaign Type
        clickid: Click ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/links/{linkid}/clicks/{clickid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_custom_field(listid: int, customfieldid: int, accountid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific custom field within an account"
    listid: List ID
        customfieldid: Custom Field ID
        accountid: Account ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/custom_fields/{customfieldid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_subscribers_for_account(ws_op: str, accountid: int, country: str=None, city: str=None, email: str=None, area_code: int=None, ad_tracking: str=None, custom_fields: str=None, latitude: int=None, last_followup_message_sent_at: str=None, longitude: int=None, name: str=None, last_followup_message_number_sent: str=None, dma_code: int=None, postal_code: str=None, misc_notes: str=None, subscribed_before: str=None, subscribed_after: str=None, region: str=None, subscribed_at: str=None, subscription_method: str=None, unsubscribed_before: str=None, tags: str=None, status: str=None, unsubscribe_method: str=None, verified_at: str=None, ws_show: str=None, ws_size: int=None, unsubscribed_after: str=None, unsubscribed_at: str=None, ws_start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the subscribers across all lists within an account"
    ws_op: Specify that you want to search
        accountid: Account ID
        country: Search for a country
        city: Search for a city
        email: Search for a subscriber's email address
        area_code: Search for an area code
        ad_tracking: Search for an ad tracking value
        custom_fields: Search for values of custom fields
        latitude: Search the latitude
        last_followup_message_sent_at: Search the date of the last follow up message
        longitude: Search the longitude
        name: Search the subscriber's name
        last_followup_message_number_sent: Search the last follow up message number sent
        dma_code: Search for a dma code
        postal_code: Search the postal code
        misc_notes: Search the miscellenous notes
        subscribed_before: Search subscribers who subscribed before a date
        subscribed_after: Search subscribers who subscriberd after a date
        region: Search the region
        subscribed_at: Search subscribers who subscribed on a date
        subscription_method: Search for subscription methods
        unsubscribed_before: Search subscribers who unsubscribed before a date
        tags: Search for a subscriber's tags
        status: Search the subscriber's status
        unsubscribe_method: Search for subscribe methods
        verified_at: Search subscribers who verified their email on a date
        ws_show: Return only the total number of rows
        ws_size: Number of items to retrieve from pagination
        unsubscribed_after: Search subscribers who unsubscribed after a date
        unsubscribed_at: Search subscribers who unsubscribed on a date
        ws_start: Starting position for pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}"
    querystring = {'ws.op': ws_op, }
    if country:
        querystring['country'] = country
    if city:
        querystring['city'] = city
    if email:
        querystring['email'] = email
    if area_code:
        querystring['area_code'] = area_code
    if ad_tracking:
        querystring['ad_tracking'] = ad_tracking
    if custom_fields:
        querystring['custom_fields'] = custom_fields
    if latitude:
        querystring['latitude'] = latitude
    if last_followup_message_sent_at:
        querystring['last_followup_message_sent_at'] = last_followup_message_sent_at
    if longitude:
        querystring['longitude'] = longitude
    if name:
        querystring['name'] = name
    if last_followup_message_number_sent:
        querystring['last_followup_message_number_sent'] = last_followup_message_number_sent
    if dma_code:
        querystring['dma_code'] = dma_code
    if postal_code:
        querystring['postal_code'] = postal_code
    if misc_notes:
        querystring['misc_notes'] = misc_notes
    if subscribed_before:
        querystring['subscribed_before'] = subscribed_before
    if subscribed_after:
        querystring['subscribed_after'] = subscribed_after
    if region:
        querystring['region'] = region
    if subscribed_at:
        querystring['subscribed_at'] = subscribed_at
    if subscription_method:
        querystring['subscription_method'] = subscription_method
    if unsubscribed_before:
        querystring['unsubscribed_before'] = unsubscribed_before
    if tags:
        querystring['tags'] = tags
    if status:
        querystring['status'] = status
    if unsubscribe_method:
        querystring['unsubscribe_method'] = unsubscribe_method
    if verified_at:
        querystring['verified_at'] = verified_at
    if ws_show:
        querystring['ws.show'] = ws_show
    if ws_size:
        querystring['ws.size'] = ws_size
    if unsubscribed_after:
        querystring['unsubscribed_after'] = unsubscribed_after
    if unsubscribed_at:
        querystring['unsubscribed_at'] = unsubscribed_at
    if ws_start:
        querystring['ws.start'] = ws_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tags_for_list(accountid: int, listid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the tags in a list"
    accountid: Account ID
        listid: LIst ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_campaigns(accountid: int, listid: int, ws_size: int=None, ws_start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the follow up and broadcasts within a list"
    accountid: Account ID
        listid: List ID
        ws_size: Number of items to retrieve from pagination
        ws_start: Starting position for pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns"
    querystring = {}
    if ws_size:
        querystring['ws.size'] = ws_size
    if ws_start:
        querystring['ws.start'] = ws_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_messages(listid: int, campaignid: int, campaigntype: str, accountid: int, ws_size: int=None, ws_start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the messages for a broadcast or follow up within a list"
    listid: List ID
        campaignid: Campaign ID
        campaigntype: Campaign Type
        accountid: Account ID
        ws_size: Number of items to retrieve from pagination
        ws_start: Starting position for pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/messages"
    querystring = {}
    if ws_size:
        querystring['ws.size'] = ws_size
    if ws_start:
        querystring['ws.start'] = ws_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_campaign(accountid: int, campaignid: int, campaigntype: str, listid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific follow-up or broadcast within an list"
    accountid: Account ID
        campaignid: Campaign ID
        campaigntype: Campaign Type
        listid: List ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_clicks(listid: int, accountid: int, campaigntype: str, linkid: int, campaignid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the clicks from a broadcast or follow up within a list"
    listid: List ID
        accountid: Account ID
        campaigntype: Campaign Type
        linkid: Link ID
        campaignid: Campaign ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/links/{linkid}/clicks"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_accounts(ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get's the accounts linked to user"
    ws_start: Starting point of the paginated collection
        ws_size: Number of items in a pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_webforms_for_account(ws_op: str, accountid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the webforms for an account"
    ws_op: Specify you want the webforms
        accountid: Account ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}"
    querystring = {'ws.op': ws_op, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_message(campaignid: int, accountid: int, messageentryid: int, campaigntype: str, listid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific message for a broadcast or follow up within a list"
    campaignid: Campaign ID
        accountid: Account ID
        messageentryid: Message Entry ID
        campaigntype: Campaign Type
        listid: List ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/messages/{messageentryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_custom_fields(listid: int, accountid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get custom fields within an account"
    listid: List ID
        accountid: Account ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/custom_fields"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_broadcast(broadcastid: int, accountid: int, listid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific broadcast within a list"
    broadcastid: Broadcast ID
        accountid: Account ID
        listid: List ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/broadcasts/{broadcastid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_segment(segmentid: int, accountid: int, listid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific segment within a list"
    segmentid: Segment ID
        accountid: Account ID
        listid: List ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/segments/{segmentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subscriber_activity(listid: int, subscriberid: int, ws_op: str, accountid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the activity for a specific subscriber"
    listid: List ID
        subscriberid: Subscriber ID
        ws_op: Specify you want the activity
        accountid: Account ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/subscribers/{subscriberid}"
    querystring = {'ws.op': ws_op, }
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_webforms_for_list(listid: int, accountid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the webforms within a list"
    listid: List ID
        accountid: Account ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/web_forms"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subscribers_for_message(campaignid: int, accountid: int, campaigntype: str, listid: int, ws_op: str, ws_start: int=None, ws_show: str=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the subscribers associated to a message for a broadcast or follow up within a list"
    campaignid: Campaign ID
        accountid: Account ID
        campaigntype: Campaign Type
        listid: List ID
        ws_op: Specify the desire to return subscribers
        ws_start: Starting position for pagination
        ws_show: Return only the total number of records
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/messages"
    querystring = {'ws.op': ws_op, }
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_show:
        querystring['ws.show'] = ws_show
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_total_broadcasts(accountid: int, listid: int, status: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get total number of broadcasts in a specific status within a list"
    accountid: Account ID
        listid: List ID
        status: Specify the status of the broadcasts to retrieve
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/broadcasts/total"
    querystring = {'status': status, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_broadcast_statistics(accountid: int, listid: int, campaignid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics for a broadcast within a list"
    accountid: Account ID
        listid: List ID
        campaignid: Campaign ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/b{campaignid}/stats"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_message_open(campaignid: int, listid: int, accountid: int, messageentryid: int, openid: int, campaigntype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific open associated to a message for a broadcast or follow up within a list"
    campaignid: Campaign ID
        listid: List ID
        accountid: Account ID
        messageentryid: Message Entry ID
        openid: Open ID
        campaigntype: Campaign Type
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/campaigns/{campaigntype}{campaignid}/messages/{messageentryid}/opens/{openid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_account(account_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific account"
    account_id: Account ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{account_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_broadcasts(accountid: int, status: str, listid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the broadcasts within a list"
    accountid: Account ID
        status: Specify the status of broadcasts to retrieve
        listid: List ID
        ws_start: Starting position for pagination
        ws_size: Number of items to retrieve from pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/broadcasts"
    querystring = {'status': status, }
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lists(accountid: int, ws_start: int=None, ws_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get's the lists for an account"
    accountid: Account ID
        ws_start: Starting point of the paginated collection
        ws_size: Number of items in a pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists"
    querystring = {}
    if ws_start:
        querystring['ws.start'] = ws_start
    if ws_size:
        querystring['ws.size'] = ws_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_split_tests_for_list(listid: int, accountid: int, ws_size: int=None, ws_start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the web form split tests within a list"
    listid: List ID
        accountid: Account ID
        ws_size: Number of items to retrieve from pagination
        ws_start: Starting position for pagination
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/web_form_split_tests"
    querystring = {}
    if ws_size:
        querystring['ws.size'] = ws_size
    if ws_start:
        querystring['ws.start'] = ws_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_split_test_for_list(splittestid: int, accountid: int, listid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific web form split test within a list"
    splittestid: Split Test ID
        accountid: Account ID
        listid: List ID
        
    """
    url = f"https://aweber2.p.rapidapi.com/1.0/accounts/{accountid}/lists/{listid}/web_form_split_tests/{splittestid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aweber2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

