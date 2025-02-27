import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def threads(authorization: str='access_token', view: str='Can be one of "id", "count", or "expanded". See Views for more info.', limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Threads are a first-class object, allowing you to build beautiful mail applications that behave the way users have come to expect. Actions like archiving or deleting can be performed on threads or individual messages.  Nylas threads messages together using a variety of heuristics. On Gmail and Microsoft Exchange accounts, messages will be threaded together as close as possible to the representation in those environments. For all other providers (including generic IMAP), messages are threaded using a custom JWZ-inspired algorithm. (Open source here, for the curious.)"
    authorization: access_token as HTTP Basic Auth username
        limit: Number of objects to return. Often defaults to 100. If set too high, requests may fail to prevent excessively large response bodies.
        
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/threads"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    if view:
        querystring['view'] = view
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Messages are the fundamental object of the Nylas platform, and the core building block for most email applications. They contain several pieces of information, such as when a message was sent, the sender's address, to whom it was sent, and the message body. They can also contain files (attachments), calendar event invitations, and more."
    
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/messages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def folders(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Folders behave like normal IMAP or filesystem folders. A Message can only exist within one folder at a time, but a Thread with many messages may span several folders.  Folders are only supported on accounts for which organization_unit is folder. You can check if an account supports labels by the organization_unit property on the Account object.  Folders support basic CRUD operations outlined in the endpoints below."
    
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/folders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def labels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Labels are equivalent to Gmail labels. Messages can have more than one label, which is popular for users who set up mail filters.  Labels are only supported on accounts for which organization_unit is label. You can check if an account supports labels by the organization_unit property on the Account object."
    
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/labels"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sending(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Nylas platform provides two ways to send messages: either through sending an existing draft, or by sending directly. Both systems send mail through the account's original SMTP/ActiveSync gateway, just as if they were sent using any other app. This means messages sent through Nylas have very high deliverability (i.e. not landing in Gmail's promotions tab), but may also be subject to backend provider rate-limiting and abuse detection. Make sure to send wisely!"
    
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/sending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendars(authorization: str='access_token', expand_recurring: str='expand_recurring', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each account connected to Nylas can have zero or more calendars, and each calendar has a collection of individual events. The calendar object is very simple, and mostly serves as a container for events. The read_only flag on a calendar indicates whether or not you can modify its properties or make changes to its events."
    expand_recurring: Using the expand_recurring URL parameter is an easy way to expand recurring events server-side so your application doesn't need to deal with RRULEs. Note that when using this query parameter, you must also use filters to specify a time range.  Currently, these expanded instances of recurring events are read-only. If the recurring event has individual modifications (overrides), such as a one-off time change, we will return these as individual events regardless of whether expand_recurring is set or not.  If expand_recurring is not set, we will return any one-off cancellations in addition to the base event, for apps that are expanding the recurrence client-side. A cancellation has the field cancelled set to true.
        
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/calendars"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    if expand_recurring:
        querystring['Expand Recurring'] = expand_recurring
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contacts(authorization: str='access_token', limit: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Nylas API provides access to the user's contacts, making it easy to add contact autocomplete, address book integration, and more to your application.  Note that contacts are currently read-only and supports both Filtering and Pagination"
    limit: Number of objects to return. Often defaults to 100. If set too high, requests may fail to prevent excessively large response bodies. See Pagination for more info.
        
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/contacts"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    if limit:
        querystring['Limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The search sub-endpoint is used to run a full-text search that is proxied to the account's provider. Results are matched with objects that have been synced, and are then returned.  The search endpoint returns 40 results by default. This endpoint supports Pagination so your application can request more objects, or iterate through all results."
    
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/search"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webhooks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Webhooks allow your application to receive notifications when certain events occur. For example, when a new email is received, Nylas will make a POST request to your URI endpoint letting you know information about the new message. You can specify what events you'd like to be notified about in the developer dashboard."
    
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/webhooks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def files(expand_recurring: str='expand_recurring', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The files endpoint manages data attached to messages. It allows you to download existing attachments from messages and threads, as well as upload new files to be sent. Note that before creating or modifying a draft to include an attachment, you must upload it via this API and use the returned file ID.  Actual attached files may be relatively large (upwards of 25MB), so this API has separate endpoints for requesting file Metadata and Downloading the actual file.  Files can be downloaded by appending /download to the file metadata URI. If available, the response will include the filename in the Content-Disposition header.  The Upload endpoint is used to transfer files to Nylas, which must be done before adding them to a draft message. Data should be sent as multipart-form data with a single field named file."
    expand_recurring: If set to true, expands single recurring events into individual event instances that fall within the requested time range.
        
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/files"
    querystring = {}
    if expand_recurring:
        querystring['Expand recurring'] = expand_recurring
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(show_cancelled: bool=None, limit: int=100, offset: int=0, event_id: str='id', calendar_id: str='id', title: str='title', description: str=None, location: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Events are objects within a calendar, generally supporting all features of modern scheduling apps. Using the calendar APIs, your application can schedule events, send meeting invitations, RSVP, and more."
    show_cancelled: If set to true, also return cancelled events. (false by default)
        limit: Number of objects to return. Often defaults to 100. If set too high, requests may fail to prevent excessively large response bodies. See Pagination for more info.
        offset: Zero-based offset from default object sorting. See Pagination for more info.
        event_id: Return the event with the given id
        calendar_id: Return events belonging to the referenced calendar
        title: Return events whose title matches the provided value.
        description: Return events whose description matches the provided value.
        location: Return events whose location matches the provided value.
        is_id: Globally unique object identifier
        
    """
    url = f"https://nylas-cloud-nylas-cloud-v1.p.rapidapi.com/events"
    querystring = {}
    if show_cancelled:
        querystring['Show cancelled'] = show_cancelled
    if limit:
        querystring['Limit'] = limit
    if offset:
        querystring['Offset'] = offset
    if event_id:
        querystring['Event ID'] = event_id
    if calendar_id:
        querystring['Calendar id'] = calendar_id
    if title:
        querystring['Title'] = title
    if description:
        querystring['Description'] = description
    if location:
        querystring['Location'] = location
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nylas-cloud-nylas-cloud-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

