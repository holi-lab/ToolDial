import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdevicefiles(size: int=20, after: str=None, page: int=None, ids: str=None, before: str=None, type: str='"image"', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search inbound files in current device"
    size: Results page size
        after: Messages created after the given date
        page: Results page number (starting from 0)
        ids: Filter files by IDs
        before: Messages created before the given date
        type: Filter files by media type
        
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/files"
    querystring = {}
    if size:
        querystring['size'] = size
    if after:
        querystring['after'] = after
    if page:
        querystring['page'] = page
    if ids:
        querystring['ids'] = ids
    if before:
        querystring['before'] = before
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getqueuestatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current device queue stats."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/queue"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicechats(type: str='"chat"', phone: str=None, broadcast: str=None, page: int=None, group: str=None, before: str=None, ids: str=None, after: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List device session chats, optionally filtered by search params."
    type: Filter messages by chat type
        phone: Filter chats by phone numbers or WIDs
        broadcast: Filter chats by broadcast list WID
        page: Results page number (starting from 0)
        group: Filter messages by group WIDs
        before: Messages created before the given date
        ids: Filter messages by chat ID
        after: Messages created after the given date
        size: Results page size
        
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/chats"
    querystring = {}
    if type:
        querystring['type'] = type
    if phone:
        querystring['phone'] = phone
    if broadcast:
        querystring['broadcast'] = broadcast
    if page:
        querystring['page'] = page
    if group:
        querystring['group'] = group
    if before:
        querystring['before'] = before
    if ids:
        querystring['ids'] = ids
    if after:
        querystring['after'] = after
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def syncdevice(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Force device status synchronization.
		
		Only one device synchronization allowed every 60 seconds."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/sync"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchfiles(ext: str=None, filename: str=None, tags: str=None, googleaccount: str=None, format: str='[
  "gif"
]', kind: str='[
  "image"
]', page: int=None, origin: str='"remote"', ids: str=None, mime: str=None, device: str=None, after: str=None, sha2: str=None, reference: str=None, filesize: str=None, size: int=20, search: str=None, status: str='[
  "any"
]', before: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List files, optionally using multiple search filters."
    format: Filter files by format type. If not present, files with any format will be returned
        ids: Filter files by IDs
        after: Deliver at after the given date
        sha2: Find a file by its SHA2 checksum value
        search: Search files by filename, tags or reference
        before: Deliver at before the given date
        
    """
    url = f"https://wassenger.p.rapidapi.com/files"
    querystring = {}
    if ext:
        querystring['ext'] = ext
    if filename:
        querystring['filename'] = filename
    if tags:
        querystring['tags'] = tags
    if googleaccount:
        querystring['googleAccount'] = googleaccount
    if format:
        querystring['format'] = format
    if kind:
        querystring['kind'] = kind
    if page:
        querystring['page'] = page
    if origin:
        querystring['origin'] = origin
    if ids:
        querystring['ids'] = ids
    if mime:
        querystring['mime'] = mime
    if device:
        querystring['device'] = device
    if after:
        querystring['after'] = after
    if sha2:
        querystring['sha2'] = sha2
    if reference:
        querystring['reference'] = reference
    if filesize:
        querystring['filesize'] = filesize
    if size:
        querystring['size'] = size
    if search:
        querystring['search'] = search
    if status:
        querystring['status'] = status
    if before:
        querystring['before'] = before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhook(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get webhook details by ID"
    
    """
    url = f"https://wassenger.p.rapidapi.com/webhooks/{webhookId}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deviceprofile(include: str='"profileImage"', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device WhatsApp profile information, including public alias, status description and profile image."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/profile"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhooks(page: int=None, size: int=20, devices: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get active webhooks, optionally filtered by search params.
		
		If you do not know what webhooks are, [read this article](https://markheath.net/post/basic-introduction-webhooks)."
    page: Results page number (starting from 0)
        size: Results page size
        devices: Filter webhooks by device IDs
        
    """
    url = f"https://wassenger.p.rapidapi.com/webhooks"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if devices:
        querystring['devices'] = devices
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicecontactdetails(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device contact details by WhatsApp ID. You can retrieve information from users and groups."
    
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/contacts/{contactWid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def syncdevicechatmessages(size: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Synchronizes device chat's messages and return them."
    size: Maximum most recent messages to sync
        
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/chats/{chatWid}/sync"
    querystring = {}
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmessage(include: str='[
  "logs"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get outbound message details."
    
    """
    url = f"https://wassenger.p.rapidapi.com/messages/{messageId}"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicebyid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device details by ID"
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdeviceinvoicepdf(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download invoice PDF document by its reference ID"
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/invoices/{invoiceId}/pdf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchmessages(devices: str=None, webhookstatus: str='[
  "invalid"
]', ids: str=None, waids: str=None, before: str=None, device: str=None, search: str=None, group: str=None, include: str='[
  "device"
]', status: str='[
  "any"
]', after: str=None, phone: str=None, priority: str='[
  "low"
]', deliverystatus: str='[
  "sent"
]', size: int=20, page: int=None, reference: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search messages, optionally filtered by search params."
    waids: Search messages by WhatsApp ID. Multiple IDs allowed separated by comma.
        before: Deliver at before the given date
        device: Search messages by origin device ID or device phone number
        search: Search messages by target phone, group, broadcast or message ID
        after: Deliver at after the given date
        size: Results page size
        page: Results page number (starting from 0)
        
    """
    url = f"https://wassenger.p.rapidapi.com/messages"
    querystring = {}
    if devices:
        querystring['devices'] = devices
    if webhookstatus:
        querystring['webhookStatus'] = webhookstatus
    if ids:
        querystring['ids'] = ids
    if waids:
        querystring['waIds'] = waids
    if before:
        querystring['before'] = before
    if device:
        querystring['device'] = device
    if search:
        querystring['search'] = search
    if group:
        querystring['group'] = group
    if include:
        querystring['include'] = include
    if status:
        querystring['status'] = status
    if after:
        querystring['after'] = after
    if phone:
        querystring['phone'] = phone
    if priority:
        querystring['priority'] = priority
    if deliverystatus:
        querystring['deliveryStatus'] = deliverystatus
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if reference:
        querystring['reference'] = reference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchdevices(size: int=20, page: int=None, status: str='any', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List user devices, optionally filtered by search params."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices"
    querystring = {}
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicefiledetails(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device file details by ID."
    
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/files/{fileId}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download file content based on its MIME type."
    
    """
    url = f"https://wassenger.p.rapidapi.com/files/{fileId}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdeviceinvoices(all: bool=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device invoices of current and past billing periods.
		
		Invoices can also be viewed as HTML and PDF."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/invoices"
    querystring = {}
    if all:
        querystring['all'] = all
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def devicehealth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current device health status.
		
		Possible status are:
		
		- **new** - Device was recently created and not authorized yet.
		
		- **error** - Device failed multiple times and is not currently available.
		
		- **loading** - Device session is loading and not ready yet.
		
		- **authorize** - Device session needs authorization via QR code.
		
		- **authorizing** - Device session is authorizing.
		
		- **offline** - Device is offline and cannot be reached via network. This is typically an operational issue.
		
		- **online** - Device is online and authorized."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicecontacts(broadcast: str=None, page: int=None, after: str=None, ids: str=None, before: str=None, type: str='"user"', phone: str=None, group: str=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List device contacts, optionally filtered by search params."
    broadcast: Filter contacts by broadlist WIDs
        page: Results page number (starting from 0)
        after: Contacts created after the given date
        ids: Filter contacts by chat ID
        before: Contacts created before the given date
        type: Filter contacts by chat type
        phone: Filter contacts by phone numbers or WIDs
        group: Filter contacts by group WIDs
        size: Results page size
        
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/contacts"
    querystring = {}
    if broadcast:
        querystring['broadcast'] = broadcast
    if page:
        querystring['page'] = page
    if after:
        querystring['after'] = after
    if ids:
        querystring['ids'] = ids
    if before:
        querystring['before'] = before
    if type:
        querystring['type'] = type
    if phone:
        querystring['phone'] = phone
    if group:
        querystring['group'] = group
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get file details looking by file ID"
    
    """
    url = f"https://wassenger.p.rapidapi.com/files/{fileId}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicechatdetails(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device chat details by WhatsApp ID. You can retrieve information from both users and groups."
    
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/chats/{chatWid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scanqr(force: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get scan QR image to create session
		
		If the device is already authenticated, it would return an error unless you pass the query param `force=true`.
		
		
		
		**Note**: this endpoint might sometimes return an error the first time the QR code is requested. Simply retry the request until you get a valid response status."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/scan"
    querystring = {}
    if force:
        querystring['force'] = force
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def previewfile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download file preview image, if available in the given file."
    
    """
    url = f"https://wassenger.p.rapidapi.com/files/{fileId}/preview"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpricing(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve plans pricing metadata"
    
    """
    url = f"https://wassenger.p.rapidapi.com/pricing"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deviceprofileimage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device WhatsApp account profile image in JPEG format. Only authorized devices can return the profile image."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/profile/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloaddevicefiledetails(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download device file contents by ID."
    
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/files/{fileId}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicegroupchats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device group chats and broadcast lists where the WhastApp account is member of."
    
    """
    url = f"https://wassenger.p.rapidapi.com/devices/{deviceId}/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicemessagedetails(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get device message details by ID."
    
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/messages/{messageWid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdevicemessages(ack: str='[
  "sent"
]', end: str=None, before: str=None, type: str='"notification"', is_from: str=None, size: int=20, flow: str='"in"', page: int=None, begin: str=None, to: str=None, after: str=None, chat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search device messages, optionally filtered by search params."
    ack: Filter messages by delivery ACK status
        end: Return oldest messages starting from the given message ID
        before: Messages created before the given date
        type: Filter messages by chat type
        is_from: Return messages sent from a given phone number or WhatsApp WID
        size: Results page size
        flow: Filter by inbound or outbound messages
        page: Results page number (starting from 0)
        begin: Return newest messages starting from the given message ID
        to: Return messages received to the given phone number or WhatsApp WID
        after: Messages created after the given date
        chat: Filter messages by chat ID
        
    """
    url = f"https://wassenger.p.rapidapi.com/io/{deviceId}/messages"
    querystring = {}
    if ack:
        querystring['ack'] = ack
    if end:
        querystring['end'] = end
    if before:
        querystring['before'] = before
    if type:
        querystring['type'] = type
    if is_from:
        querystring['from'] = is_from
    if size:
        querystring['size'] = size
    if flow:
        querystring['flow'] = flow
    if page:
        querystring['page'] = page
    if begin:
        querystring['begin'] = begin
    if to:
        querystring['to'] = to
    if after:
        querystring['after'] = after
    if chat:
        querystring['chat'] = chat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wassenger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

