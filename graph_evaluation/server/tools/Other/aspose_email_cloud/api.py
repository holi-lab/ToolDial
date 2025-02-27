import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ainameexpand(name: str, language: str=None, script: str=None, encoding: str=None, style: str='0', location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: A name to expand.
        language: An ISO-639 code of the language; either 639-1 or 639-3 (e.g. "it" or "ita"
for Italian).
            
        script: A writing system code; starts with the ISO-15924 script name.
        encoding: A character encoding name.
        style: Name writing style.
Enum, available values: Formal, Informal, Legal, Academic
        location: A geographic code such as an ISO-3166 two letter country code, for example
"FR" for France.
            
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/AiName/expand"
    querystring = {'name': name, }
    if language:
        querystring['language'] = language
    if script:
        querystring['script'] = script
    if encoding:
        querystring['encoding'] = encoding
    if style:
        querystring['style'] = style
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def storageexists(storagename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    storagename: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contactgetlist(format: str, folder: str='folder/on/storage', pagenumber: int=0, itemsperpage: int=10, storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Contact document format.
Enum, available values: VCard, WebDav, Msg
        folder: Path to folder in storage.
        pagenumber: Page number.
        itemsperpage: Count of items on page.
        storage: Storage name.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/Contact/list"
    querystring = {'format': format, }
    if folder:
        querystring['folder'] = folder
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if itemsperpage:
        querystring['itemsPerPage'] = itemsperpage
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendargetasalternate(calendaraction: str, filename: str, sequenceid: str=None, folder: str='calendar/location/on/storage', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    calendaraction: iCalendar method type
Enum, available values: Create, Update, Cancel
        filename: iCalendar file name in storage
        sequenceid: The sequence id
        folder: Path to folder in storage
        storage: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/Calendar/as-alternate"
    querystring = {'calendarAction': calendaraction, 'fileName': filename, }
    if sequenceid:
        querystring['sequenceId'] = sequenceid
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendargetlist(folder: str, storage: str='First Storage', itemsperpage: int=10, pagenumber: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    folder: Path to folder in storage.
        storage: Storage name.
        itemsperpage: Count of items on page.
        pagenumber: Page number.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/Calendar/list"
    querystring = {'folder': folder, }
    if storage:
        querystring['storage'] = storage
    if itemsperpage:
        querystring['itemsPerPage'] = itemsperpage
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientthreadgetmessages(account: str, threadid: str, storage: str='First Storage', folder: str='INBOX', accountstoragefolder: str='email/account/location/on/storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    account: Email account
        threadid: Thread identifier
        storage: Storage name where account file located
        folder: Specifies account folder to get thread from
            
        accountstoragefolder: Folder in storage where account file located
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/thread/messages"
    querystring = {'account': account, 'threadId': threadid, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if accountstoragefolder:
        querystring['accountStorageFolder'] = accountstoragefolder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendargetasfile(filename: str, format: str, folder: str='calendar/file/location/on/storage', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: Calendar document file name.
        format: File format.
Enum, available values: Ics, Msg
        folder: Path to folder in storage.
        storage: Storage name.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/Calendar/as-file"
    querystring = {'fileName': filename, 'format': format, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def disposableemailisdisposable(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    address: An email address to check
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/disposable/is-disposable"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailgetlist(format: str, folder: str='folder/on/storage', storage: str='First Storage', pagenumber: int=0, itemsperpage: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Email document format.
Enum, available values: Eml, Msg, MsgUnicode, Mhtml, Html, Tnef, Oft
        folder: Path to folder in storage.
        storage: Storage name.
        pagenumber: Page number.
        itemsperpage: Count of items on page.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/list"
    querystring = {'format': format, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if itemsperpage:
        querystring['itemsPerPage'] = itemsperpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendarget(filename: str, folder: str='calendar/location/on/storage', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: iCalendar file name in storage.
        folder: Path to folder in storage.
        storage: Storage name.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/Calendar"
    querystring = {'fileName': filename, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientaccountgetmulti(filename: str, storage: str='First Storage', folder: str='email/account/location/on/storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: File name on storage
        storage: Storage name
        folder: Folder on storage
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/account/multi"
    querystring = {'fileName': filename, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailgetasfile(format: str, filename: str, folder: str='folder/on/storage', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: File format
Enum, available values: Eml, Msg, MsgUnicode, Mhtml, Html, Tnef, Oft
        filename: Email document file name
        folder: Path to folder in storage
        storage: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/as-file"
    querystring = {'format': format, 'fileName': filename, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contactget(filename: str, format: str, storage: str='First Storage', folder: str='folder/on/storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: Contact document file name.
        format: Contact document format.
Enum, available values: VCard, WebDav, Msg
        storage: Storage name.
        folder: Path to folder in storage.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/Contact"
    querystring = {'fileName': filename, 'format': format, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientaccountget(filename: str, folder: str='email/account/location/on/storage', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: File name on storage.
        folder: Folder on storage.
        storage: Storage name.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/account"
    querystring = {'fileName': filename, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailconfigdiscover(address: str, fastprocessing: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    address: Email address.
        fastprocessing: Turns on fast processing. All discover systems will run in parallel.
First discovered result will be returned.
            
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/config/discover"
    querystring = {'address': address, }
    if fastprocessing:
        querystring['fastProcessing'] = fastprocessing
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contactgetasfile(filename: str, fromformat: str, toformat: str, folder: str='folder/on/storage', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: Calendar document file name
        fromformat: File format to convert from
Enum, available values: VCard, WebDav, Msg
        toformat: File format
Enum, available values: VCard, WebDav, Msg
        folder: Path to folder in storage
        storage: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/Contact/as-file"
    querystring = {'fileName': filename, 'fromFormat': fromformat, 'toFormat': toformat, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mapimessageget(format: str, filename: str, storage: str='First Storage', folder: str='folder/on/storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Email document format.
Enum, available values: Eml, Msg, MsgUnicode, Mhtml, Html, Tnef, Oft
        filename: Email document file name.
        storage: Storage name.
        folder: Path to folder in storage.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/MapiMessage"
    querystring = {'format': format, 'fileName': filename, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailget(format: str, filename: str, storage: str='First Storage', folder: str='folder/on/storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Email document format.
Enum, available values: Eml, Msg, MsgUnicode, Mhtml, Html, Tnef, Oft
        filename: Email document file name.
        storage: Storage name.
        folder: Path to folder in storage.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email"
    querystring = {'format': format, 'fileName': filename, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientmessagefetch(account: str, messageid: str, format: str='Eml', type: str='Dto', accountstoragefolder: str='email/account/location/on/storage', storage: str='First Storage', folder: str='INBOX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    account: Email account
        messageid: Message identifier
        format: Base64 data format. Used only if type is set to Base64.
Enum, available values: Eml, Msg, MsgUnicode, Mhtml, Html, Tnef, Oft
        type: MailMessageBase type. Using this property you can fetch message in different
formats (as EmailDto, MapiMessageDto or a file represented as Base64 string).
            
Enum, available values: Dto, Mapi, Base64
        accountstoragefolder: Folder in storage where account file located.
        storage: Storage name where account file located.
        folder: Account folder to fetch from (should be specified for some protocols such as
IMAP)
            
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/message"
    querystring = {'account': account, 'messageId': messageid, }
    if format:
        querystring['format'] = format
    if type:
        querystring['type'] = type
    if accountstoragefolder:
        querystring['accountStorageFolder'] = accountstoragefolder
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mapicontactget(format: str, filename: str, storage: str='First Storage', folder: str='folder/on/storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Contact document format.
Enum, available values: VCard, WebDav, Msg
        filename: Contact document file name.
        storage: Storage name.
        folder: Path to folder in storage.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/MapiContact"
    querystring = {'format': format, 'fileName': filename, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ainamegenderize(name: str, encoding: str=None, language: str=None, location: str=None, style: str='0', script: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: A name to genderize.
        encoding: A character encoding name.
        language: An ISO-639 code of the language; either 639-1 or 639-3 (e.g. "it" or "ita"
for Italian).
            
        location: A geographic code such as an ISO-3166 two letter country code, for example
"FR" for France.
            
        style: Name writing style.
Enum, available values: Formal, Informal, Legal, Academic
        script: A writing system code; starts with the ISO-15924 script name.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/AiName/genderize"
    querystring = {'name': name, }
    if encoding:
        querystring['encoding'] = encoding
    if language:
        querystring['language'] = language
    if location:
        querystring['location'] = location
    if style:
        querystring['style'] = style
    if script:
        querystring['script'] = script
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientmessagelist(account: str, folder: str, recursive: bool=None, accountstoragefolder: str='email/account/location/on/storage', type: str='Dto', storage: str='First Storage', format: str='Eml', querystring: str="('From' Contains '.com')", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The query string should have the following view.
		
		    The example of a simple expression:
		
		
		    '<Field name>' <Comparison operator> '<Field value>',
		
		where
		&lt;Field Name&gt; - the name of a message field through which filtering is made,
		&lt;Comparison operator&gt; - comparison operators, as their name implies, allow to compare
		message field and
		specified value,
		&lt;Field value&gt; - value to be compared with a message field.
		
		    The number of simple expressions can make a compound one, ex.:
		    (<Simple expression 1> & <Simple expression 2>) | <Simple expression 3
		    >,
		
		where
		"&amp;" - logical-AND operator,
		"|" - logical-OR operator
		
		    At present the following values are allowed as a field name (<Field name>):
		
		"To" - represents a TO field of message,
		"Text" - represents string in the header or body of the message,
		"Bcc" - represents a BCC field of message,
		"Body" - represents a string in the body of message,
		"Cc" - represents a CC field of message,
		"From" - represents a From field of message,
		"Subject" - represents a string in the subject of message,
		"InternalDate" - represents an internal date of message,
		"SentDate" - represents a sent date of message
		
		    Additionally, the following field names are allowed for IMAP-protocol:
		
		"Answered" - represents an /Answered flag of message
		"Seen" - represents a /Seen flag of message
		"Flagged" - represents a /Flagged flag of message
		"Draft" - represents a /Draft flag of message
		"Deleted" - represents a Deleted/ flag of message
		"Recent" - represents a Deleted/ flag of message
		"MessageSize" - represents a size (in bytes) of message
		
		    Additionally, the following field names are allowed for Exchange:
		
		"IsRead" - Indicates whether the message has been read
		"HasAttachment" - Indicates whether or not the message has attachments
		"IsSubmitted" - Indicates whether the message has been submitted to the Outbox
		"ContentClass" - represents a content class of item
		
		    Additionally, the following field names are allowed for pst/ost files:
		
		"MessageClass" - Represents a message class
		"ContainerClass" - Represents a folder container class
		"Importance" - Represents a message importance
		"MessageSize" - represents a size (in bytes) of message
		"FolderName" - represents a folder name
		"ContentsCount" - represents a total number of items in the folder
		"UnreadContentsCount" - represents the number of unread items in the folder.
		"Subfolders" - Indicates whether or not the folder has subfolders
		"Read" - the message is marked as having been read
		"HasAttachment" - the message has at least one attachment
		"Unsent" - the message is still being composed
		"Unmodified" - the message has not been modified since it was first saved (if unsent) or it was
		delivered (if sent)
		"FromMe" - the user receiving the message was also the user who sent the message
		"Resend" - the message includes a request for a resend operation with a non-delivery report
		"NotifyRead" - the user who sent the message has requested notification when a recipient first
		reads it
		"NotifyUnread" - the user who sent the message has requested notification when a recipient
		deletes it before
		reading or the Message object expires
		"EverRead" - the message has been read at least once
		
		    The field value (<Field value>) can take the following values:
		    For text fields - any string,
		    For date type fields - the string of "d-MMM-yyy" format, ex. "10-Feb-2009",
		    For flags (fields of boolean type) - either "True", or "False"
		
		            "
    account: Email account
        folder: A folder in email account
        recursive: Specifies that should message be searched in subfolders recursively
        accountstoragefolder: Folder in storage where account file located
        type: MailMessageBase type. Using this property you can get messages in different
formats (as EmailDto, MapiMessageDto or a file represented as Base64 string).
            
Enum, available values: Dto, Mapi, Base64
        storage: Storage name where account file located
        format: Base64 data format. Used only if type is set to Base64.
Enum, available values: Eml, Msg, MsgUnicode, Mhtml, Html, Tnef, Oft
        querystring: A MailQuery search string
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/message/list"
    querystring = {'account': account, 'folder': folder, }
    if recursive:
        querystring['recursive'] = recursive
    if accountstoragefolder:
        querystring['accountStorageFolder'] = accountstoragefolder
    if type:
        querystring['type'] = type
    if storage:
        querystring['storage'] = storage
    if format:
        querystring['format'] = format
    if querystring:
        querystring['queryString'] = querystring
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ainameparseemailaddress(emailaddress: str, language: str=None, encoding: str=None, style: str='0', script: str=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    emailaddress: Email address to parse.
        language: An ISO-639 code of the language; either 639-1 or 639-3 (e.g. "it" or "ita"
for Italian).
            
        encoding: A character encoding name.
        style: Name writing style.
Enum, available values: Formal, Informal, Legal, Academic
        script: A writing system code; starts with the ISO-15924 script name.
        location: A geographic code such as an ISO-3166 two letter country code, for example
"FR" for France.
            
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/AiName/parse-email-address"
    querystring = {'emailAddress': emailaddress, }
    if language:
        querystring['language'] = language
    if encoding:
        querystring['encoding'] = encoding
    if style:
        querystring['style'] = style
    if script:
        querystring['script'] = script
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ainamecomplete(name: str, script: str=None, encoding: str=None, language: str=None, location: str=None, style: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: A name to complete.
        script: A writing system code; starts with the ISO-15924 script name.
        encoding: A character encoding name.
        language: An ISO-639 code of the language; either 639-1 or 639-3 (e.g. "it" or "ita"
for Italian).
            
        location: A geographic code such as an ISO-3166 two letter country code, for example
"FR" for France.
            
        style: Name writing style.
Enum, available values: Formal, Informal, Legal, Academic
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/AiName/complete"
    querystring = {'name': name, }
    if script:
        querystring['script'] = script
    if encoding:
        querystring['encoding'] = encoding
    if language:
        querystring['language'] = language
    if location:
        querystring['location'] = location
    if style:
        querystring['style'] = style
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfile(path: str, versionid: str=None, storagename: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/folder/file.ext'
        versionid: File version ID to download
        storagename: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/storage/file/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileversions(path: str, storagename: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/file.ext'
        storagename: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ainamematch(name: str, othername: str, language: str=None, script: str=None, encoding: str=None, location: str=None, style: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: A name to match.
        othername: Another name to match.
        language: An ISO-639 code of the language; either 639-1 or 639-3 (e.g. "it" or "ita"
for Italian).
            
        script: A writing system code; starts with the ISO-15924 script name.
        encoding: A character encoding name.
        location: A geographic code such as an ISO-3166 two letter country code, for example
"FR" for France.
            
        style: Name writing style.
Enum, available values: Formal, Informal, Legal, Academic
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/AiName/match"
    querystring = {'name': name, 'otherName': othername, }
    if language:
        querystring['language'] = language
    if script:
        querystring['script'] = script
    if encoding:
        querystring['encoding'] = encoding
    if location:
        querystring['location'] = location
    if style:
        querystring['style'] = style
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientthreadgetlist(account: str, folder: str, accountstoragefolder: str='email/account/location/on/storage', storage: str='First Storage', messagescachelimit: int=200, updatefoldercache: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    account: Email account
        folder: A folder in email account.
            
        accountstoragefolder: Folder in storage where account file located
        storage: Storage name where account file located
        messagescachelimit: Limit messages cache size if CacheFile is used. Ignored in
accounts without limits support
            
        updatefoldercache: This parameter is only used in accounts with CacheFile.
If true - get new messages and update threads cache for given folder.
If false, get only threads from cache without any calls to an email account
            
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/thread/list"
    querystring = {'account': account, 'folder': folder, }
    if accountstoragefolder:
        querystring['accountStorageFolder'] = accountstoragefolder
    if storage:
        querystring['storage'] = storage
    if messagescachelimit:
        querystring['messagesCacheLimit'] = messagescachelimit
    if updatefoldercache:
        querystring['updateFolderCache'] = updatefoldercache
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientfoldergetlist(account: str, accountstoragefolder: str='email/account/location/on/storage', parentfolder: str='INBOX', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    account: Email account
        accountstoragefolder: Folder in storage where account file located
        parentfolder: Folder in which subfolders should be listed
        storage: Storage name where account file located
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/folder/list"
    querystring = {'account': account, }
    if accountstoragefolder:
        querystring['accountStorageFolder'] = accountstoragefolder
    if parentfolder:
        querystring['parentFolder'] = parentfolder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mapicalendarget(filename: str, folder: str='calendar/location/on/storage', storage: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filename: Calendar file name in storage.
        folder: Path to folder in storage.
        storage: Storage name.
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/MapiCalendar"
    querystring = {'fileName': filename, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientmessagefetchfile(account: str, messageid: str, format: str='Eml', storage: str='First Storage', accountstoragefolder: str='email/account/location/on/storage', folder: str='INBOX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    account: Email account
        messageid: Message identifier
        format: Fetched message file format.
Enum, available values: Eml, Msg, MsgUnicode, Mhtml, Html, Tnef, Oft
        storage: Storage name where account file located.
        accountstoragefolder: Folder in storage where account file located.
        folder: Account folder to fetch from (should be specified for some protocols such as
IMAP)
            
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/client/message/file"
    querystring = {'account': account, 'messageId': messageid, }
    if format:
        querystring['format'] = format
    if storage:
        querystring['storage'] = storage
    if accountstoragefolder:
        querystring['accountStorageFolder'] = accountstoragefolder
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ainameformat(name: str, location: str=None, style: str='0', format: str='%t%L%f%m', script: str=None, encoding: str=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: A name to format.
        location: A geographic code such as an ISO-3166 two letter country code, for example
"FR" for France.
            
        style: Name writing style.
Enum, available values: Formal, Informal, Legal, Academic
        format: Format of the name. Predefined format can be used by ID, or custom format can be specified.
Predefined formats:

    /format/default/ (= '%t%F%m%N%L%p')
    /format/FN+LN/ (= '%F%L')
    /format/title+FN+LN/ (= '%t%F%L')
    /format/FN+MN+LN/ (= '%F%M%N%L')
    /format/title+FN+MN+LN/ (= '%t%F%M%N%L')
    /format/FN+MI+LN/ (= '%F%m%N%L')
    /format/title+FN+MI+LN/ (= '%t%F%m%N%L')
    /format/LN/ (= '%L')
    /format/title+LN/ (= '%t%L')
    /format/LN+FN+MN/ (= '%L,%F%M%N')
    /format/LN+title+FN+MN/ (= '%L,%t%F%M%N')
    /format/LN+FN+MI/ (= '%L,%F%m%N')
    /format/LN+title+FN+MI/ (= '%L,%t%F%m%N')

Custom format string - custom combination of characters and the next term placeholders:

    '%t' - Title (prefix)
    '%F' - First name
    '%f' - First initial
    '%M' - Middle name(s)
    '%m' - Middle initial(s)
    '%N' - Nickname
    '%L' - Last name
    '%l' - Last initial
    '%p' - Postfix

If no value for format option was provided, its default value is '%t%F%m%N%L%p'
            
        script: A writing system code; starts with the ISO-15924 script name.
        encoding: A character encoding name.
        language: An ISO-639 code of the language; either 639-1 or 639-3 (e.g. "it" or "ita"
for Italian).
            
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/AiName/format"
    querystring = {'name': name, }
    if location:
        querystring['location'] = location
    if style:
        querystring['style'] = style
    if format:
        querystring['format'] = format
    if script:
        querystring['script'] = script
    if encoding:
        querystring['encoding'] = encoding
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileslist(path: str, storagename: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: Folder path e.g. '/folder'
        storagename: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdiscusage(storagename: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    storagename: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ainameparse(name: str, language: str='eng', script: str=None, encoding: str=None, location: str='USA', style: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: A name to parse.
        language: An ISO-639 code of the language; either 639-1 or 639-3 (e.g. "it" or "ita"
for Italian).
            
        script: A writing system code; starts with the ISO-15924 script name.
        encoding: A character encoding name.
        location: A geographic code such as an ISO-3166 two letter country code, for example
"FR" for France.
            
        style: Name writing style.
Enum, available values: Formal, Informal, Legal, Academic
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/AiName/parse"
    querystring = {'name': name, }
    if language:
        querystring['language'] = language
    if script:
        querystring['script'] = script
    if encoding:
        querystring['encoding'] = encoding
    if location:
        querystring['location'] = location
    if style:
        querystring['style'] = style
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def objectexists(path: str, versionid: str=None, storagename: str='First Storage', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File or folder path e.g. '/file.ext' or '/folder'
        versionid: File version ID
        storagename: Storage name
        
    """
    url = f"https://aspose-email-cloud.p.rapidapi.com/email/storage/exist/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-email-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

