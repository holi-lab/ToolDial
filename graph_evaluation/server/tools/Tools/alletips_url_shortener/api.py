import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_existing_links(publickey: str, domain: str='alle.tips', url: str='alle.tips/abcdef', created: str=None, inactive: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all short links you created within the defined parameters."
    created: Shows links that were created at or after this date
        inactive: Shows also deactivated links
        
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/link/"
    querystring = {'PUBLICKEY': publickey, }
    if domain:
        querystring['domain'] = domain
    if url:
        querystring['url'] = url
    if created:
        querystring['created'] = created
    if inactive:
        querystring['inactive'] = inactive
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_click_counts_per_shortlink(publickey: str, campaignid: str=None, enddate: str=None, sort: str=None, startdate: str=None, domain: str=None, inactive: bool=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of your links and the individual click count."
    campaignid: Limits the results to all links within this campaign.
        enddate: Limits results to links with a enddate lesser or equal to the specified one.
        sort: Orders the shortlinks by creation date.
(asc = ascending, desc = descending)
        startdate: Limits results to links with a startdate greater or equal to the specified one.
        domain: Limits results to a specific domain (e.g. alle.tips)
        inactive: Returns also stats for inactive links.
        limit: Limits the number of returned stats.
        
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/linkstats/"
    querystring = {'PUBLICKEY': publickey, }
    if campaignid:
        querystring['campaignid'] = campaignid
    if enddate:
        querystring['enddate'] = enddate
    if sort:
        querystring['sort'] = sort
    if startdate:
        querystring['startdate'] = startdate
    if domain:
        querystring['domain'] = domain
    if inactive:
        querystring['inactive'] = inactive
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_click_history_for_campaign(publickey: str, campaignid: str, startdate: str=None, enddate: str=None, link: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the number of clicks (total, link clicks and QR-clicks) for a specific campaign.
		Additionally an array with chart data will be returned (x: date, y: total clicks)."
    campaignid: Limits the results to a specific campaign.
        startdate: Returns events at and after this date.
        enddate: Returns events at and before this date.
        link: Returns values for a specific link.
        
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/clicks/"
    querystring = {'PUBLICKEY': publickey, 'campaignid': campaignid, }
    if startdate:
        querystring['startdate'] = startdate
    if enddate:
        querystring['enddate'] = enddate
    if link:
        querystring['link'] = link
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_number_of_active_links(publickey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the number of active links for your accounts."
    
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/linkcount/"
    querystring = {'PUBLICKEY': publickey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_click_history(publickey: str, enddate: str=None, link: str='alle.tips/rapidapi', startdate: str=None, campaignid: str='abcd-efgh-ijk9', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the number of clicks (total, link clicks and QR-clicks).
		Additionally an array with chart data will be returned (x: date, y: total clicks)."
    enddate: Returns events at and before this date.
        link: Returns values for a specific link.
        startdate: Returns events at and after this date.
        campaignid: Limits the results to a specific campaign.
        
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/clicks/"
    querystring = {'PUBLICKEY': publickey, }
    if enddate:
        querystring['enddate'] = enddate
    if link:
        querystring['link'] = link
    if startdate:
        querystring['startdate'] = startdate
    if campaignid:
        querystring['campaignid'] = campaignid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_qr_code_for_a_shortlink(publickey: str, link: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a QR (quick response) code for your shortlink.
		Scans of this code will be counted as regular clicks, but also reported separately."
    link: Your shortlink to be QRed (without https:// etc.)
        
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/qr/"
    querystring = {'PUBLICKEY': publickey, 'link': link, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_existing_campaigns(publickey: str, nocampaign: bool=None, name: str='some campaign name', startdate: str=None, enddate: str=None, inactive: bool=None, campaignid: str='abcd-efgh-ijk2', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all your campaigns (= groups of links)."
    nocampaign: Returns also a summary of the shortlinks that haven't been assigned to a campaign.
        name: Get all campaigns thats title includes [name]
        startdate: Shows all campaigns with a startdate greater or equal this date.
        enddate: Shows all campaigns with an enddate greater or equal this date (and an earlier startdate).
        inactive: Returns also inactive campaigns.
Default: only active ones.
        campaignid: Get all infos for this specific campaign
        limit: Limits the number of results
        
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/campaign/"
    querystring = {'PUBLICKEY': publickey, }
    if nocampaign:
        querystring['nocampaign'] = nocampaign
    if name:
        querystring['name'] = name
    if startdate:
        querystring['startdate'] = startdate
    if enddate:
        querystring['enddate'] = enddate
    if inactive:
        querystring['inactive'] = inactive
    if campaignid:
        querystring['campaignid'] = campaignid
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_available_domains(publickey: str, favorite: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of domains you can use to shorten your urls."
    favorite: Show default domain for this/your account
        
    """
    url = f"https://alletips-url-shortener.p.rapidapi.com/domains"
    querystring = {'PUBLICKEY': publickey, }
    if favorite:
        querystring['favorite'] = favorite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alletips-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

