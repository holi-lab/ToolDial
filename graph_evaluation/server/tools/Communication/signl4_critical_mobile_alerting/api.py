import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def alerts_alertid(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets an alert by id."
    alertid: Id of the requested Alert.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/alerts/{alertid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_userid_image(userid: str, width: int=None, height: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/users/{userid}/image"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_memberships(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: Team ID of team you want to request.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/memberships"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_userid(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a user object with details such as his email address and duty information."
    userid: Identifier of the user to get. Use 'Me' to get information about the currently logged in user. This is not possible with an api key.
Can also be an email address of a user in the team or the unique id of an according user object.”
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/users/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_dutysummary(teamid: str, lasttwoduties: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of the team the duty belongs to.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/dutysummary"
    querystring = {}
    if lasttwoduties:
        querystring['lastTwoDuties'] = lasttwoduties
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scripts_inventory(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all available inventory scripts which can be added to a SIGNL4 subscription."
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/scripts/inventory"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_teamid_metrics(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sample Request:
		            
		    GET /categories/cbb70402-1359-477f-ac92-0171cf2b5ff7/metrics"
    teamid: ID of the team the categories belongs to
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/categories/{teamid}/metrics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prepaid_balance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/prepaid/balance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_setupprogress(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of the team the progress should be retrieved for.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/setupProgress"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_dutyreports_filename(filename: str, teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: Filename of the csv to download.
        teamid: ID of team you want to download the duty report for.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/dutyReports/{filename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_alertid_attachments_attachmentid(attachmentid: str, alertid: str, height: int=None, scale: bool=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    attachmentid: Id of the attachment, that you want to retrieve.
        alertid: Id of the alert that contains the wanted attachment.
        height: Optional parameter defining the wanted height of the picture that is retrieved.
        scale: Optional parameter defining whether it's wanted to scale the retrieved image. Default is set to
true.
        width: Optional parameter defining the wanted width of the picture that is retrieved.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/alerts/{alertid}/attachments/{attachmentid}"
    querystring = {}
    if height:
        querystring['height'] = height
    if scale:
        querystring['scale'] = scale
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid_channelprices(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    subscriptionid: Id of the subscription that needs to be retrieved.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}/channelPrices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_alertid_annotations(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get annotations of an alert by id."
    alertid: Id of the requested Alert.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/alerts/{alertid}/annotations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_schedules_scheduleid(teamid: str, scheduleid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: Id of the team the duty belongs to
        scheduleid: Id of the requested duty schedule.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/schedules/{scheduleid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prepaid_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/prepaid/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_userid_setupprogress(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    userid: ID of the user the progress should be retrieved for.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/users/{userid}/setupProgress"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_eventid_parameters(eventid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get parameters of an event by id."
    eventid: Event Id of the requested Alert.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/events/{eventid}/parameters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of user objects with details such as their email address and duty information. Only users who
		are part of your team will be returned."
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_eventid_overview(eventid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overview event by id."
    eventid: Id of event to get.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/events/{eventid}/overview"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webhooks(teamid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of defined outbound webhooks in the system."
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/webhooks"
    querystring = {}
    if teamid:
        querystring['teamId'] = teamid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_schedules(teamid: str, userid: str=None, limit: int=None, mindate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: Id of the team the schedules user belongs to
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/schedules"
    querystring = {}
    if userid:
        querystring['UserId'] = userid
    if limit:
        querystring['Limit'] = limit
    if mindate:
        querystring['MinDate'] = mindate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_alertreports_filename(teamid: str, filename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of team you want to get the duty report file infos for.
        filename: File name of file you want to download.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/alertReports/{filename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_teamid_categoryid_subscriptions(categoryid: str, teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sample Request:
		            
		    GET /categories/cbb70402-1359-477f-ac92-0171cf2b5ff7/c0054336-cd89-4abf-882d-ba69b5adb25e/subscriptions
		    {
		    }"
    categoryid: Category to get subscriptions for
        teamid: ID of the team the category belongs to
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/categories/{teamid}/{categoryid}/subscriptions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_alertsettings(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of the team the settings should be retrieved for.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/alertSettings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_images(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/categories/images"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scripts_instances_instanceid(instanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the script instance specified by the passed instance id."
    instanceid: Instance Id of script instance to be returned.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/scripts/instances/{instanceid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of the team that should be retrieved.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scripts_inventory_parsed(language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/scripts/inventory/parsed"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_teamid_categoryid_metrics(categoryid: str, teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sample Request:
		            
		    GET /categories/cbb70402-1359-477f-ac92-0171cf2b5ff7/c0054336-cd89-4abf-882d-ba69b5adb25e/metrics"
    categoryid: ID of the category to get
        teamid: ID of the team the category belongs to
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/categories/{teamid}/{categoryid}/metrics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid_userlicenses(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    subscriptionid: ID of the subscription
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}/userLicenses"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_teamid_categoryid(categoryid: str, teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sample Request:
		            
		    GET /categories/cbb70402-1359-477f-ac92-0171cf2b5ff7/c0054336-cd89-4abf-882d-ba69b5adb25e"
    categoryid: ID of the category to get
        teamid: ID of the team the category belongs to
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/categories/{teamid}/{categoryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid_prepaidsettings(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}/prepaidSettings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid_prepaidbalance(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}/prepaidBalance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_alertid_overview(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overview alert by id."
    alertid: Id of the requested Alert.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/alerts/{alertid}/overview"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scripts_instances(teamid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all script instances in the subscription."
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/scripts/instances"
    querystring = {}
    if teamid:
        querystring['teamId'] = teamid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_teamid(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sample Request:
		            
		    GET /categories/cbb70402-1359-477f-ac92-0171cf2b5ff7"
    teamid: ID of the team the categories belong to
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/categories/{teamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scripts_inventory_parsed_scriptid(scriptid: str, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the script specified by the passed script id."
    scriptid: The Id of the script to be returned.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/scripts/inventory/parsed/{scriptid}"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid_prepaidtransactions(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    subscriptionid: ID of the subscription to get transactions for
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}/prepaidTransactions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prepaid_transactions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/prepaid/transactions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_alertreports(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of team you want to download reports from.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/alertReports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    subscriptionid: Id of the subscription that's to be retrieved.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_eventsources(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of the team the sources should be retrieved for.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/eventSources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid_features(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    subscriptionid: Id of the subscription from which the features need to be retrieved.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}/features"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_alertid_attachments(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get attachments of an alert by id."
    alertid: Id of the requested Alert.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/alerts/{alertid}/attachments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscriptions_subscriptionid_teams(subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/subscriptions/{subscriptionid}/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_report(userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about the occurred alert volume in different time periods as well as information about the
		response behaviour (amount of confirmed and unhandled alerts) of your team members."
    userid: User ID of the user for whom you want a report.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/alerts/report"
    querystring = {}
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_userid_dutystatus(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a object with duty information."
    userid: Identifier of the user to get. Use 'Me' to get information about the currently logged in user. This is not possible with an api key.
Can also be an email address of a user in the team or the unique id of an according user object.”
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/users/{userid}/dutyStatus"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_alertid_notifications(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get notifications of all users by alert id."
    alertid: Id of the requested Alert.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/alerts/{alertid}/notifications"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_teamid_dutyreports(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    teamid: ID of team you want to get the duty report file infos for.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/teams/{teamid}/dutyReports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhookbyid(webhookid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information of the webhook specified by the passed id."
    webhookid: Id of the outbound webhook to be retrieved.
        
    """
    url = f"https://signl4-critical-mobile-alerting.p.rapidapi.com/webhooks/{webhookid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signl4-critical-mobile-alerting.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

