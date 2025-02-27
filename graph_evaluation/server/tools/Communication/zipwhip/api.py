import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_save(session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Save details about the user associated with the provided session"
    session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/user/save"
    querystring = {'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_save(name: str, type: str, session: str, textline: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a group. A named group is a unique set of contacts that are referenced similar to a contact  A group is defined as device:/xxxxxxxxxx/xx. After the :/, this is the phone number associated with the account you are sending messages under. The /xx is the id for the group."
    name: This is the name of the group that you would like to associate with a group of contacts when using group/save.
        type: This is used to identify the type of group you wish to create. For now, please use 'Group'
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        textline: If the group is being created within a landline or toll-free number than this parameter is required. This is the routing information for sending out the group message. Example format textline:/xxxxxxxxxx?carrier=xxx. After the :/ this is the logged in number's phone number. The carrier is equal to the carrier returned from user/get.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/group/save"
    querystring = {'name': name, 'type': type, 'session': session, }
    if textline:
        querystring['textline'] = textline
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_addmember(address: str, group: str, session: str, city: str=None, email: str=None, firstname: str=None, lastname: str=None, loc: str=None, notes: str=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Once a group is created via the group/save call, you then need to populate members."
    address: For US domestic use 10-digit number. For International numbers use full E.164 format. Examples: US: 5555555555 E.164: +1155555555555
        group: The address that identifies the group in the logged in account. Format will be similar to: device:/xxxxxxxxxx/xx.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        city: The name city where the contact resides.
        email: The email address of the contact being saved.
        firstname: The first name of the contact being saved.
        lastname: The last name of the contact being saved.
        loc: Notes about the contact, limit 255 characters.
        notes: Notes about the contact, limit 255 characters.
        state: The state where the contact resides.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/group/addmember"
    querystring = {'address': address, 'group': group, 'session': session, }
    if city:
        querystring['city'] = city
    if email:
        querystring['email'] = email
    if firstname:
        querystring['firstName'] = firstname
    if lastname:
        querystring['lastName'] = lastname
    if loc:
        querystring['loc'] = loc
    if notes:
        querystring['notes'] = notes
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def message_delete(message: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Marks the given message as deleted in Zipwhip's database."
    message: The message ID being modified. In all cases the method will take multiple messages as parameters.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/message/delete"
    querystring = {'message': message, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contact_delete(contact: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method can take one or more contact ID(s) and delete those records from Zipwhip. A contact's ID is available from contact/list or part of the return from message/send."
    contact: This is the contact ID that is return from contact/list. A contact ID is also part of the return from message/send.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/contact/delete"
    querystring = {'contact': contact, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def message_list(session: str, limit: str=None, start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves messages for account. A start of 0 provides newest to oldest messages."
    session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        limit: Limit the number of elements returned when paging.
        start: When paging through content, this field is used to tell the controller what element to start at for the next page.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/message/list"
    querystring = {'session': session, }
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_login(username: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes a username and password and returns json with the account's session key. A sessionkey lasts forever, so it is best to store the sessionkey and use it for future purposes."
    username: For US domestic use 10-digit number. For International numbers use full E.164 format. Examples: US: 5555555555 E.164: +1155555555555
        password: Password associated with phone number being entered in user/login.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/user/login"
    querystring = {'username': username, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_get(name: str, type: str, session: str, textline: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive details about an existing group. This call can be used to review the members of the group."
    name: This is the name of the group that you would like to associate with a group of contacts when using group/save.
        type: This is used to identify the type of group you wish to create. For now, please use 'Group'
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        textline: If the group is being created within a landline or toll-free number than this parameter is required. This is the routing information for sending out the group message. Example format textline:/xxxxxxxxxx?carrier=xxx. After the :/ this is the logged in number's phone number. The carrier is equal to the carrier returned from user/get.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/group/get"
    querystring = {'name': name, 'type': type, 'session': session, }
    if textline:
        querystring['textline'] = textline
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_removemember(group: str, member: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove a group member. This is useful if a group member has requested to opt-out of notifications."
    group: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        member: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/group/removemember"
    querystring = {'group': group, 'member': member, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def message_read(messages: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Marks the given message as read in Zipwhip's database."
    messages: The message ID being modified. In all cases the method will take multiple messages as parameters.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/message/read"
    querystring = {'messages': messages, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversation_delete(fingerprint: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Deletes the conversation based on the provided conversation ID. It also deletes the associated messages."
    fingerprint: The unique identifier for a conversation. This can be retrieved from a conversation/list.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/conversation/delete"
    querystring = {'fingerprint': fingerprint, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contact_save(address: str, session: str, city: str=None, email: str=None, firstname: str=None, lastname: str=None, loc: str=None, notes: str=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Save details about the contact for the given phone number."
    address: For US domestic use 10-digit number. For International numbers use full E.164 format. Examples: US: 5555555555 E.164: +1155555555555
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        city: The name city where the contact resides.
        email: The email address of the contact being saved.
        firstname: The first name of the contact being saved.
        lastname: Notes about the contact, limit 255 characters.
        loc: Notes about the contact, limit 255 characters.
        notes: Notes about the contact, limit 255 characters.
        state: The state where the contact resides.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/contact/save"
    querystring = {'address': address, 'session': session, }
    if city:
        querystring['city'] = city
    if email:
        querystring['email'] = email
    if firstname:
        querystring['firstName'] = firstname
    if lastname:
        querystring['lastName'] = lastname
    if loc:
        querystring['loc'] = loc
    if notes:
        querystring['notes'] = notes
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def message_send(contacts: str, body: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sends a message from the logged in user's phone number."
    contacts: This is the contact ID that is return from contact/list. A contact ID is also part of the return from message/send.
        body: The message body that you would like to send. This is a maximum of 160 ASCII characters or 140 Bytes.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/message/send"
    querystring = {'contacts': contacts, 'body': body, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversation_list(fingerprint: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Deletes the conversation based on the provided conversation ID. It also deletes the associated messages."
    fingerprint: The unique identifier for a conversation. This can be retrieved from a conversation/list.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/conversation/list"
    querystring = {'fingerprint': fingerprint, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contact_list(session: str, page: str=None, pagesize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method returns all contacts associated with session provided."
    session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        page: For paginated results iterate over the requested page(s) for all available content.
        pagesize: Specify the number of results returned for a paginated method.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/contact/list"
    querystring = {'session': session, }
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_delete(address: str, session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete the group. This is similar to deleting a contact, no message objects will be affected by deleting the group."
    address: For US domestic use 10-digit number. For International numbers use full E.164 format. Examples: US: 5555555555 E.164: +1155555555555
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/group/delete"
    querystring = {'address': address, 'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversation_get(fingerprint: str, session: str, limit: str, start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves conversation details and messages that specific to the conversation ID provided."
    fingerprint: The unique identifier for a conversation. This can be retrieved from a conversation/list.
        session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        limit: Limit the number of elements returned when paging.
        start: When paging through content, this field is used to tell the controller what element to start at for the next page.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/conversation/get"
    querystring = {'fingerprint': fingerprint, 'session': session, 'limit': limit, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_get(session: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides information about the logged in user or account associated with sessionKey passed in."
    session: This is the sessionKey that is returned from user/login. sessionKeys do not expire. Because of this, it is recommended that you do a single user/login and then save the sessionKey locally for future web calls.
        
    """
    url = f"https://community-zipwhip.p.rapidapi.com/user/get"
    querystring = {'session': session, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zipwhip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

