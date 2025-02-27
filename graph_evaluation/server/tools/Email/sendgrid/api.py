import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_a_specific_block(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a specific email address from your blocks list. Blocks happen when your message was rejected for a reason related to the message, not the recipient address. This can happen when your mail server IP address has been added to a blacklist or blocked by an ISP, or if the message content is flagged by a filter on the receiving server."
    email: The email address of the specific block.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/blocks/{email}}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_blocks(limit: int=None, offset: int=None, start_time: int=None, end_time: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all email addresses that are currently on your blocks list.  There are several causes for  emails: for example, your mail server IP address is on an ISP blacklist, or blocked by an ISP, or if the receiving server flags the message content."
    limit: Limit the number of results to be displayed per page.
        offset: The point in the list to begin displaying results.
        start_time: Refers start of the time range in unix timestamp when a blocked email was created (inclusive). Ex/ 1514764800
        end_time: Refers end of the time range in unix timestamp when a blocked email was created (inclusive). Ex/ 1514764800
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/blocks"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_google_analytics_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current setting for Google Analytics."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/tracking_settings/google_analytics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_bounce(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a specific bounce for a given email address.  A bounced email is when the message is undeliverable and then returned to the server that sent it."
    email: The email address you would like to remove from the bounce list.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/bounces/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_bounces(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve all of your bounces. A bounced email is when the message is undeliverable and then returned to the server that sent it."
    limit: Optional field to limit the number of results returned. If not used, then 500 is the default limit.
        offset: Optional beginning point in the list to retrieve from.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/bounces"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_invalid_emails(limit: int=None, offset: int=None, start_time: str=None, end_time: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all invalid email addresses.  An invalid email occurs when you attempt to send email to an address that is formatted in a manner that does not meet internet email format standards or the email does not exist at the recipient’s mail server.  Examples include addresses without the “@” sign or addresses that include certain special characters and/or spaces. This response can come from our own server or the recipient mail server."
    limit: Limit the number of results to be displayed per page. Example: 10
        offset: Paging offset. The point in the list to begin displaying results. Example: 0
        start_time: Refers start of the time range in unix timestamp when an invalid email was created (inclusive).
        end_time: Refers end of the time range in unix timestamp when an invalid email was created (inclusive).
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/invalid_emails"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_specific_invalid_email(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a specific invalid email addresses.  An invalid email occurs when you attempt to send email to an address that is formatted in a manner that does not meet internet email format standards or the email does not exist at the recipient’s mail server.  Examples include addresses without the “@” sign or addresses that include certain special characters and/or spaces. This response can come from our own server or the recipient mail server."
    email: The specific email address of the invalid email entry that you want to retrieve.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/invalid_emails/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_spam_reports(start_time: int=None, end_time: int=None, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve all spam reports. Spam reports happen when a recipient indicates that they think your email is spam and then their email provider reports this to SendGrid."
    start_time: Refers start of the time range in unix timestamp when a spam report was created (inclusive).
        end_time: Refers end of the time range in unix timestamp when a spam report was created (inclusive).
        limit: Limit the number of results to be displayed per page. Example: 10
        offset: Paging offset. The point in the list to begin displaying results. Example: 0
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/spam_reports"
    querystring = {}
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_specific_spam_report(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a specific spam report. Spam reports happen when a recipient indicates that they think your email is spam and then their email provider reports this to SendGrid."
    email: The email address of a specific spam report that you want to retrieve.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/suppression/spam_reports/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_batch_id(batch_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    batch_id: The batch ID you want to get. Example: HkJ5yLYULb7Rj8GKSx7u025ouWVlMgAi
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail/batch/{batch_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_scheduled_sends(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve all cancel/paused scheduled send information.  The Cancel Scheduled Sends feature allows the customer to cancel a scheduled send based on a Batch ID included in the SMTPAPI header. Scheduled sends cancelled less than 10 minutes before the scheduled time are not guaranteed to be cancelled."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/user/scheduled_sends"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_scheduled_send(batch_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve the cancel/paused scheduled send information for a specific batch_id.  The Cancel Scheduled Sends feature allows the customer to cancel a scheduled send based on a Batch ID included in the SMTPAPI header. Scheduled sends cancelled less than 10 minutes before the scheduled time are not guaranteed to be cancelled."
    batch_id: The batch ID you want to retrieve. Example: HkJ5yLYULb7Rj8GKSx7u025ouWVlMgAi
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/user/scheduled_sends/{batch_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_suppression_groups_associated_with_the_user(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all suppression groups created by this user.  Suppression groups, or unsubscribe groups, are specific types or categories of email that you would like your recipients to be able to unsubscribe from. For example: Daily Newsletters, Invoices, System Alerts.  The name and description of the unsubscribe group will be visible by recipients when they are managing their subscriptions.  Each user can create up to 25 different suppression groups."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/asm/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_information_on_a_single_suppression_group(group_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a single suppression group.  Suppression groups, or unsubscribe groups, are specific types or categories of email that you would like your recipients to be able to unsubscribe from. For example: Daily Newsletters, Invoices, System Alerts.  The name and description of the unsubscribe group will be visible by recipients when they are managing their subscriptions.  Each user can create up to 25 different suppression groups."
    group_id: The ID of the suppression group you would like to retrieve.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/asm/groups/{group_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_suppressions_for_a_suppression_group(group_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve all suppressed email addresses belonging to the given group.  Suppressions are recipient email addresses that are added to unsubscribe groups. Once a recipient's address is on the suppressions list for an unsubscribe group, they will not receive any emails that are tagged with that unsubscribe group."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/asm/groups/{group_id}/suppressions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_suppression_groups_for_an_email_address(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the list of all groups that the given email address has been unsubscribed from.  Suppressions are a list of email addresses that will not receive content sent under a given group."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/asm/suppressions/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_tracking_settings(limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all tracking settings that you can enable on your account.  You can track a variety of the actions your recipients may take when interacting with your emails including opening your emails, clicking on links in your emails, and subscribing to (or unsubscribing from) your emails.  For more information about tracking, please see our User Guide."
    limit: The number of settings to return.
        offset: Where in the list of results you want to begin retrieving settings.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/tracking_settings"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_click_track_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current click tracking setting.  You can track a variety of the actions your recipients may take when interacting with your emails including opening your emails, clicking on links in your emails, and subscribing to (or unsubscribing from) your emails.  For more information about tracking, please see our User Guide."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/tracking_settings/click"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_suppressions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all suppressions. Suppressions are a list of email addresses that will not receive content sent under a given group."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/asm/suppressions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_subscription_tracking_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current settings for subscription tracking.  Subscription tracking adds links to the bottom of your emails that allows your recipients to subscribe to, or unsubscribe from, your emails.  You can track a variety of the actions your recipients may take when interacting with your emails including opening your emails, clicking on links in your emails, and subscribing to (or unsubscribing from) your emails."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/tracking_settings/subscription"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_open_tracking_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current settings for open tracking.  Open Tracking adds an invisible image at the end of the email which can track email opens. If the email recipient has images enabled on their email client, a request to SendGrid’s server for the invisible image is executed and an open event is logged. These events are logged in the Statistics portal, Email Activity interface, and are reported by the Event Webhook.  You can track a variety of the actions your recipients may take when interacting with your emails including opening your emails, clicking on links in your emails, and subscribing to (or unsubscribing from) your emails."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/tracking_settings/open"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_mail_settings(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all mail settings.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    offset: Where in the list of results to begin displaying settings.
        limit: The number of settings to return.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_bcc_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current BCC mail settings.  When the BCC mail setting is enabled, SendGrid will automatically send a blind carbon copy (BCC) to an address for every email sent without adding that address to the header. Please note that only one email address may be entered in this field, if you wish to distribute BCCs to multiple addresses you will need to create a distribution group or use forwarding rules.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/bcc"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_address_whitelist_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current email address whitelist settings.  The address whitelist setting whitelists a specified email address or domain for which mail should never be suppressed. For example, you own the domain “example.com,” and one or more of your recipients use email@example.com addresses, by placing example.com in the address whitelist setting, all bounces, blocks, and unsubscribes logged for that domain will be ignored and sent as if under normal sending conditions.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s  Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/address_whitelist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_footer_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current Footer mail settings.  The footer setting will insert a custom footer at the bottom of the text and HTML bodies. Use the embedded HTML editor and plain text entry fields to create the content of the footers to be inserted into your emails.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/footer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_forward_spam_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current Forward Spam mail settings.  Enabling the forward spam setting allows you to specify an email address to which spam reports will be forwarded.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/forward_spam"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_plain_content_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current Plain Content mail settings.  The plain content setting will automatically convert any plain text emails that you send to HTML before sending.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/plain_content"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_spam_check_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current Spam Checker mail settings.  The spam checker filter notifies you when emails are detected that exceed a predefined spam threshold.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/spam_check"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_bounce_purge_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current bounce purge settings.  This setting allows you to set a schedule for SendGrid to automatically delete contacts from your soft and hard bounce suppression lists.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/bounce_purge"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_forward_bounce_mail_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your current bounce forwarding mail settings.  Activating this setting allows you to specify an email address to which bounce reports are forwarded.  Mail settings allow you to tell SendGrid specific things to do to every email that you send to your recipients over SendGrid’s Web API."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mail_settings/forward_bounce"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_parse_settings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve all of your current inbound parse settings.  The inbound parse webhook allows you to have incoming emails parsed, extracting some or all of the contnet, and then have that content POSTed by SendGrid to a URL of your choosing."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/user/webhooks/parse/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_specific_parse_setting(hostname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a specific inbound parse setting.  The inbound parse webhook allows you to have incoming emails parsed, extracting some or all of the contnet, and then have that content POSTed by SendGrid to a URL of your choosing."
    hostname: The hostname associated with the inbound parse setting that you would like to retrieve.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/user/webhooks/parse/settings/{hostname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_email_statistics_by_country_and_state_province(start_date: str, limit: int=None, offset: int=None, aggregated_by: str=None, end_date: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your email statistics segmented by country and state/province.  We only store up to 7 days of email activity in our database. By default, 500 items will be returned per request via the Advanced Stats API endpoints.  Advanced Stats provide a more in-depth view of your email statistics and the actions taken by your recipients. You can segment these statistics by geographic location, device type, client type, browser, and mailbox provider."
    start_date: The starting date of the statistics to retrieve. Must be in format YYYY-MM-DD
        limit: How many results to include on each page.
        offset: How many results to exclude.
        aggregated_by: How you would like the statistics to be grouped. Must be either "day", "week", or "month".  Allowed Values: day, week, month
        end_date: The end date of the statistics to retrieve.  default: The date the request is made.
        country: The country you would like to see statistics for. Currently only supported for US and CA.  Allowed Values: US, CA
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/geo/stats"
    querystring = {'start_date': start_date, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if aggregated_by:
        querystring['aggregated_by'] = aggregated_by
    if end_date:
        querystring['end_date'] = end_date
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_global_email_statistics(start_date: str, limit: int=None, offset: int=None, aggregated_by: str=None, end_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve all of your global email statistics between a given date range."
    start_date: The starting date of the statistics to retrieve. Must follow format YYYY-MM-DD.
        limit: The number of results to return.
        offset: The point in the list to begin retrieving results.
        aggregated_by: How to group the statistics. Must be either "day", "week", or "month".
        end_date: The end date of the statistics to retrieve. Defaults to today. Must follow format YYYY-MM-DD.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/stats"
    querystring = {'start_date': start_date, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if aggregated_by:
        querystring['aggregated_by'] = aggregated_by
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_email_statistics_by_device_type(start_date: str, end_date: str=None, aggregated_by: str=None, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your email statistics segmented by the device type.  We only store up to 7 days of email activity in our database. By default, 500 items will be returned per request via the Advanced Stats API endpoints."
    start_date: The starting date of the statistics to retrieve.
        end_date: The end date of the statistics to retrieve. Defaults to today.
        aggregated_by: How to group the statistics. Must be either "day", "week", or "month".
        limit: How many results to include on each page.
        offset: How many results to exclude.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/devices/stats"
    querystring = {'start_date': start_date, }
    if end_date:
        querystring['end_date'] = end_date
    if aggregated_by:
        querystring['aggregated_by'] = aggregated_by
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_email_statistics_by_client_type(start_date: str, end_date: str=None, aggregated_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your email statistics segmented by client type.  We only store up to 7 days of email activity in our database. By default, 500 items will be returned per request via the Advanced Stats API endpoints.  Advanced Stats provide a more in-depth view of your email statistics and the actions taken by your recipients. You can segment these statistics by geographic location, device type, client type, browser, and mailbox provider."
    start_date: The starting date of the statistics to retrieve. Must follow format YYYY-MM-DD.
        end_date: The end date of the statistics to retrieve. Defaults to today. Must follow format YYYY-MM-DD.
        aggregated_by: How to group the statistics. Must be either "day", "week", or "month".  Allowed Values: day, week, month
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/clients/stats"
    querystring = {'start_date': start_date, }
    if end_date:
        querystring['end_date'] = end_date
    if aggregated_by:
        querystring['aggregated_by'] = aggregated_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_stats_by_a_specific_client_type(start_date: str, client_type: str, end_date: str=None, aggregated_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your email statistics segmented by a specific client type.  We only store up to 7 days of email activity in our database. By default, 500 items will be returned per request via the Advanced Stats API endpoints.  Available Client Types phone tablet webmail desktop Advanced Stats provide a more in-depth view of your email statistics and the actions taken by your recipients. You can segment these statistics by geographic location, device type, client type, browser, and mailbox provider."
    start_date: The starting date of the statistics to retrieve. Must follow format YYYY-MM-DD.
        client_type: Specifies the type of client to retrieve stats for. Must be either "phone", "tablet", "webmail", or "desktop".
        end_date: The end date of the statistics to retrieve. Defaults to today. Must follow format YYYY-MM-DD.
        aggregated_by: How to group the statistics. Must be either "day", "week", or "month".  Allowed Values: day, week, month
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/clients/{client_type}/stats"
    querystring = {'start_date': start_date, }
    if end_date:
        querystring['end_date'] = end_date
    if aggregated_by:
        querystring['aggregated_by'] = aggregated_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_email_statistics_by_mailbox_provider(start_date: str, limit: int=None, offset: int=None, aggregated_by: str=None, end_date: str=None, mailbox_providers: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your email statistics segmented by recipient mailbox provider.  We only store up to 7 days of email activity in our database. By default, 500 items will be returned per request via the Advanced Stats API endpoints.  Advanced Stats provide a more in-depth view of your email statistics and the actions taken by your recipients. You can segment these statistics by geographic location, device type, client type, browser, and mailbox provider."
    start_date: The starting date of the statistics to retrieve. Must follow format YYYY-MM-DD.
        limit: The number of results to include on each page.
        offset: The number of results to exclude.
        aggregated_by: How to group the stats. Must be either "day", "wee", or "month".  Allowed Values: day, week, month
        end_date: The end date of the statistics to retrieve. Defaults to today. Must follow format YYYY-MM-DD.
        mailbox_providers: The mail box providers to get statistics for. You can include up to 10 by including this parameter multiple times.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/mailbox_providers/stats"
    querystring = {'start_date': start_date, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if aggregated_by:
        querystring['aggregated_by'] = aggregated_by
    if end_date:
        querystring['end_date'] = end_date
    if mailbox_providers:
        querystring['mailbox_providers'] = mailbox_providers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_email_statistics_by_browser(start_date: str, end_date: str=None, limit: int=None, offset: int=None, aggregated_by: str=None, browsers: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve your email statistics segmented by browser type.  We only store up to 7 days of email activity in our database. By default, 500 items will be returned per request via the Advanced Stats API endpoints.  Advanced Stats provide a more in-depth view of your email statistics and the actions taken by your recipients. You can segment these statistics by geographic location, device type, client type, browser, and mailbox provider."
    start_date: The starting date of the statistics to retrieve. Must follow format YYYY-MM-DD.
        end_date: The end date of the statistics to retrieve. Defaults to today.
        limit: The number of results to include on each page.
        offset: The number of results to exclude.
        aggregated_by: How to group the stats. Must be either "day", "week", or "month".  Allowed Values: day, week, month
        browsers: The browsers to get statistics for. You can include up to 10 different browsers by including this parameter multiple times.
        
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/browsers/stats"
    querystring = {'start_date': start_date, }
    if end_date:
        querystring['end_date'] = end_date
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if aggregated_by:
        querystring['aggregated_by'] = aggregated_by
    if browsers:
        querystring['browsers'] = browsers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_alerts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retieve all of your alerts."
    
    """
    url = f"https://rapidprod-sendgrid-v1.p.rapidapi.com/alerts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

