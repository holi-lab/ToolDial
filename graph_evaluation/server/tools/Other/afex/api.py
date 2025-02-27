import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def jobtitles(authorization: str, api_key: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of job titles for use with [api/v1/CorporateClient](https://doc.api.afex.com/?version=latest#02991c42-6ae4-45a3-ac54-46e2f77b0a6c) and [api/v1/PrivateClient](https://doc.api.afex.com/?version=latest#993c336c-c9d2-4572-bcce-540bba557530)."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/jobTitles"
    querystring = {'Authorization': authorization, 'API-Key': api_key, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobtitles(authorization: str, api_key: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of job titles for use with [api/v1/CorporateClient](https://doc.api.afex.com/?version=latest#02991c42-6ae4-45a3-ac54-46e2f77b0a6c) and [api/v1/PrivateClient](https://doc.api.afex.com/?version=latest#993c336c-c9d2-4572-bcce-540bba557530)."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/jobTitles"
    querystring = {'Authorization': authorization, 'API-Key': api_key, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findcountry(content_type: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all countries including for each country, the Country Name and Country Code.  For use when calling the following methods:
		
		1. [api/BeneficiaryCreate](https://doc.api.afex.com/?version=latest#dce44b8e-5798-472c-80ac-161aaf2f70d7)
		2. [api/BeneficiaryUpdate](https://doc.api.afex.com/?version=latest#e3a7d23e-f7b2-4789-873a-b6ccf38e6b15)"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/country/find"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def beneficiaryinformationanderrorcodes(api_key: str, content_type: str, information: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all response codes related to:
		
		1. [api/BeneficiaryCreate](https://doc.api.afex.com/?version=latest#dce44b8e-5798-472c-80ac-161aaf2f70d7)
		2. [api/BeneficiaryUpdate](https://doc.api.afex.com/?version=latest#e3a7d23e-f7b2-4789-873a-b6ccf38e6b15)
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Information|'True' returns informative codes; 'False' returns error codes.|**True**|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/beneficiaryCodes"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Information': information, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def identification_types_contact(api_key: str, content_type: str, authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of identification types for use with [api/v1/CorporateClient](https://doc.api.afex.com/?version=latest#02991c42-6ae4-45a3-ac54-46e2f77b0a6c) and [api/v1/PrivateClient](https://doc.api.afex.com/?version=latest#993c336c-c9d2-4572-bcce-540bba557530)."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/primaryContactIdentificationType"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companytypes(content_type: str, api_key: str, authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of company types for use with [api/v1/CorporateClient](https://doc.api.afex.com/?version=latest#02991c42-6ae4-45a3-ac54-46e2f77b0a6c) and [api/v1/PrivateClient](https://doc.api.afex.com/?version=latest#993c336c-c9d2-4572-bcce-540bba557530)."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/companyType"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def naicscodes(content_type: str, authorization: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of NAICS (North American Industry Classification System) codes for use with [api/v1/CorporateClient](https://doc.api.afex.com/?version=latest#02991c42-6ae4-45a3-ac54-46e2f77b0a6c) and [api/v1/PrivateClient](https://doc.api.afex.com/?version=latest#993c336c-c9d2-4572-bcce-540bba557530)."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/naics"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, 'API-Key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def naicscodes(content_type: str, authorization: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of NAICS (North American Industry Classification System) codes for use with [api/v1/CorporateClient](https://doc.api.afex.com/?version=latest#02991c42-6ae4-45a3-ac54-46e2f77b0a6c) and [api/v1/PrivateClient](https://doc.api.afex.com/?version=latest#993c336c-c9d2-4572-bcce-540bba557530)."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/naics"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, 'API-Key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getquote(api_key: str, content_type: str, amount: str, currencypair: str, valuedate: str, isamountsettlement: str, optiondate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a rate quote for a specified currency pair and value date (note, next available value dates are returned by [api/ValueDates](https://doc.api.afex.com/?version=latest#f8a2ef3b-0ed0-478f-9a03-9cf5337c9177).
		
		The quote ID returned may then be supplied when calling either [api/Trades/Create](https://doc.api.afex.com/?version=latest#91dda474-6ac2-435c-8999-e2f9302130c6) or [api/Forwards/Create](https://doc.api.afex.com/?version=latest#de38bc9f-dce5-4d58-89ec-e81ff50b7da3) to fix the rate for a trade at the quoted rate (providing the quote has not yet expired).  A quote is valid for 30 seconds.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Amount|The Trade (buy) amount of the transaction. May be set as Settlement (sell) amount if 'True' is passed in [IsAmountSettlement] parameter.|**True**|
		|CurrencyPair|The currency pair specifies the trade (buy) and settlement (sell) currencies being quoted. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair. For example, if the client wants to convert AUD to USD, the currency pair would be expressed as USDAUD in the quote.|**True**|
		|IsAmountSettlement|Passing "True" will set 'Amount' to settlement (sell) currency.|False|
		|OptionDate|If the quote is for a window / flex forward, the OptionDate must be supplied in the [api/Forwards/Create](https://doc.api.afex.com/?version=latest#de38bc9f-dce5-4d58-89ec-e81ff50b7da3). The option date is the start date of the forward window period.|True *for a window forward only*|
		|ValueDate|The date the proceeds of the trade are paid out (providing the client has remitted settlement funds to AFEX).|**True**|"
    amount: Optional. Amount / Volume of either settlement or trade currency, depending on what is specified for **IsAmountSettlement**.   Where the client has pricing tiers configured, the **Amount** is used to place the transaction in a pricing tier and apply the appropriate rate.
        currencypair: Required.  3 character ISO currency codes for the buy and sell curencies. The currency pair specifies the trade (buy) and settlement (sell) currencies. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair e.g USDAUD for selling AUD to buy USD.
        valuedate: Required.  YYYY/MM/DD.  For a spot trade, valid value dates can be obtained via api/valuedates.  Dates beyond spot are forward dates and there is not an API method to return valid forward dates.  If the ValueDate supplied is not a valid business day, an error message will be returned
        isamountsettlement: Optional. TRUE or FALSE.  If the **Amount** is the settlement (sell) amount, this is TRUE.  If the **Amount** is the trade currency amount (buy),  **IsAmountSettlement** may be left blank as the **Amount** field will be defaulted to mean the buy currency.
        optiondate: Optional.  YYYY/MM/DD.  If quote is for a window forward an option date must be specfied.  If the date supplied is a non working day, an error message will be returned
        
    """
    url = f"https://afex2.p.rapidapi.com/api/quote"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Amount': amount, 'CurrencyPair': currencypair, 'ValueDate': valuedate, 'IsAmountSettlement': isamountsettlement, 'OptionDate': optiondate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallbeneficiaries(content_type: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of all beneficiaries including for each beneficiary, basic information and status. 
		
		Where a vendor ID is supplied as a URI parameter, the full set of beneficiary data is returned for the specified vendor ID."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/beneficiary"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forwarddetails(api_key: str, content_type: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details on a Forward by passing the 'TradeNumber' or retrieve all forward transactions by omitting 'ID' in the request.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Id|The TradeNumber (AFEX transaction ID) can be passed to retrieve details of a single transaction|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/forwards"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paymentdetailsv_2(content_type: str, api_key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details on a Payment by passing the 'TradeNumber' or retrieve all Payments by omitting 'ID' in the request.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Id|The ReferenceNumber (AFEX transaction ID) can be passed to retrieve details of a single transaction.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v2/payments"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tradedetails(content_type: str, api_key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details on a Trade by passing the 'TradeNumber' or retrieve all Trades by omitting 'ID' in the request.
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Id|The TradeNumber (AFEX transaction ID) can be passed to retrieve details of a single transaction.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/trades"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def purposeofpaymentcodes(content_type: str, api_key: str, bankcountrycode: str, beneficiarycountrycode: str, highvalue: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive a list of PurposeOfPaymentCodes.  A valid AFEX Purpose of Payment Code must be passed in RemittanceLine2.
		
		For use when calling the following methods:
		
		1. [api/BeneficiaryCreate](https://doc.api.afex.com/?version=latest#dce44b8e-5798-472c-80ac-161aaf2f70d7)
		2. [api/BeneficiaryUpdate](https://doc.api.afex.com/?version=latest#e3a7d23e-f7b2-4789-873a-b6ccf38e6b15)
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|BankCountryCode|Country codes available via [api/Country/Find](https://doc.api.afex.com/?version=latest#36bf62c1-96c8-4e1d-a51c-13fab3ace7b5).|**True**|
		|BeneficiaryCountryCode|Country codes available via [api/Country/Find](https://doc.api.afex.com/?version=latest#36bf62c1-96c8-4e1d-a51c-13fab3ace7b5).|**True**|
		|Currency|Currency of beneficiary.|**True**|
		|HighValue|High or low value beneficiary.  When not specified, default is High.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/purposeOfPayment"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'BankCountryCode': bankcountrycode, 'BeneficiaryCountryCode': beneficiarycountrycode, 'HighValue': highvalue, 'Currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfundingbalanceidandaccountid_instantpayments(api_key: str, content_type: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Funding Balance ID and an Account ID.  A funding balance is required to set up an "instant" beneficiary to receive AFEX Client to Client payments in the currency.  This data must be provided to the remitter of an AFEX "instant" client to client payment so the remitter can set up the instant beneficiary.  If a funding balance ID does not exist, one must be created using [api/FundingBalance/Create](https://doc.api.afex.com/?version=latest#e9dfa409-544b-460a-b894-e7174c1defaf).
		
		**Request Parameters**
		------------
		
		|Parameter|Description|Required|
		|---------|-----------|-----------|
		|Currency| ISO 3 letter currency code e.g. USD, AUD etc.|**True**|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/fundingbalance"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paymentdetails(api_key: str, content_type: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details on a Payment by passing the 'TradeNumber' or retrieve all Payments by omitting 'ID' in the request.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Id|The ReferenceNumber (AFEX transaction ID) can be passed to retrieve details of a single transaction.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/payments"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findcity(api_key: str, content_type: str, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all suburbs/cities by country code.
		
		For use when calling the following methods:
		
		1. [api/BeneficiaryCreate](https://doc.api.afex.com/?version=latest#dce44b8e-5798-472c-80ac-161aaf2f70d7)
		2. [api/BeneficiaryUpdate](https://doc.api.afex.com/?version=latest#e3a7d23e-f7b2-4789-873a-b6ccf38e6b15)
		
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|**CountryCode**|Country codes available via [api/Country/Find](https://doc.api.afex.com/?version=latest#36bf62c1-96c8-4e1d-a51c-13fab3ace7b5).|**True**|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/city/find"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'CountryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tradeconfirmation(api_key: str, content_type: str, tradenumber: str, swift: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve an image of a confirmation for a specified trade or a payment (SWIFT) confirmation.  This is the PDF confirmation that AFEX emails to Communication Recipients set up on the client's account.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|TradeNumber|TradeNumber (AFEX transaction ID) for the payment for which you are retrieving the confirmation.|**True**|
		|Swift|When set to "true" the SWIFT Confirmation will be returned.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/confirmations"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'TradeNumber': tradenumber, 'Swift': swift, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveprivateclientaccountdetails(content_type: str, api_key: str, authorization: str, client_apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the account account details for the specified private client.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|ApiKey|APIKey of the account which you would like the customer details.|**True**|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/privateclient"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'Authorization': authorization, 'Client_ApiKey': client_apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientaccountstatus(content_type: str, authorization: str, api_key: str, client_apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the account status of a child account by API key.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|ApiKey|API key for child account.|**True**|
		
		**Status Definitions**
		
		| Status  |  Description                            |
		|-----------|-------------------------------------------------------------------|
		|PendingApproval      | The account has been created and is waiting to be submitted to the AFEX Compliance team for approval.  An account in this status cannot trade.|
		|SubmittedforApproval      |   The account has been submitted for assessment and approval.  An account in this status cannot trade.|
		|Approved      |   The account has been approved by AFEX and is able to trade.|
		|Closed      |  The account is closed, is no longer active and cannot trade.|  
		|ReferAFEX      | All other account statuses not included in the above. The client should be referred to AFEX regarding further information on the account status.|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/clientstatus"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, 'API-Key': api_key, 'Client_ApiKey': client_apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holidays(content_type: str, api_key: str, currencypair: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all non-processing days (holidays) for a currency pair.  Data returned can be used to derive a value date beyond spot value for use when creating a forward, drawdown, payment, or predelivery. It should be noted that the api/valuedates method cannot return value dates beyond spot.  
		
		The logic for deriving a valid value date from api/holidays response is shown below.
		
		A valid value date is any date which:
		
		* is not included in the list of holidays returned for the currency pair AND 
		* falls on a Monday to Friday AND
		* If the date is to be used to create forward, is within the maximum forward tenor allowed for the client account"
    currencypair: Required. 3 character ISO currency code . The currency pair specifies the trade (buy) and settlement (sell) currencies. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair. For example if the client wants to convert AUD to USD, the currency pair would be expressed as USDAUD.
        
    """
    url = f"https://afex2.p.rapidapi.com/api/holidays"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'CurrencyPair': currencypair, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tradedetailsv_2(api_key: str, content_type: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details on a Trade by passing the 'TradeNumber' or retrieve all Trades by omitting 'ID' in the request.
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Id|The TradeNumber (AFEX transaction ID) can be passed to retrieve details of a single transaction.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v2/trades"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvaluedatev_2(content_type: str, api_key: str, currencypair: str, valuetype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The value date is the date that the trade proceeds are paid out to either the beneficiary, or in the case of a funding trade, to the client's AFEX currency holding.  Value dates must be business days for both the buy (trade) and sell (settlement) currencies.  The AFEX value date calendar is based on local business dates, business dates in the country of the trade and settlement currency and local payment cut off times by currency. 
		
		
		The api/valuedates method returns next available value date for the ValueType supplied.  ValueType can be CASH (value today), TOM (value next available business day), or SPOT (value 2 business days).  If the value type specified is not allowed for the currency due to AFEX currency rules, an error message will be returned rather than a date.  Value dates beyond spot i.e. forward value dates, are not available via the api/valuedates method.  These can be calculated using inputs from the api/holidays method response.
		
		Value dates returned by the [api/valuedates](https://doc.api.afex.com/?version=latest#f8a2ef3b-0ed0-478f-9a03-9cf5337c9177) call can be used to populate the ValueDate field in the following calls: [api/trades/create](https://doc.api.afex.com/?version=latest#91dda474-6ac2-435c-8999-e2f9302130c6), [api/payments/create](https://doc.api.afex.com/?version=latest#9a8d48e2-012e-4f05-8f43-bb238cf705da), [api/drawdown/create](https://doc.api.afex.com/?version=latest#db41f516-bf89-405c-9cb5-e320ce27fd16), and [api/predelivery/create](https://doc.api.afex.com/?version=latest#5b4dfe12-abe9-4f23-9c07-d3eeabd0debf).
		 
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|CurrencyPair|3-character ISO currency codes for the buy and sell currencies. The currency pair specifies the trade (buy) and settlement (sell) currencies. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair e.g. USDAUD for selling AUD to buy USD.|**True**|
		|ValueType|CASH, TOM or SPOT.  CASH=today, TOM=next business day,  SPOT=2 business days.  If a ValueType is not specified, the SPOT value type will be defaulted.   If the value type is not allowed for the currency provided, an error message will be returned.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v2/valuedates"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'CurrencyPair': currencypair, 'ValueType': valuetype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getquotev_2(api_key: str, content_type: str, isamountsettlement: str, optiondate: str, valuedate: str, currencypair: str, amount: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a rate quote for a specified currency pair and value date (note, next available value dates are returned by [api/ValueDates](https://doc.api.afex.com/?version=latest#f8a2ef3b-0ed0-478f-9a03-9cf5337c9177).
		
		The quote ID returned may then be supplied when calling either [api/Trades/Create](https://doc.api.afex.com/?version=latest#91dda474-6ac2-435c-8999-e2f9302130c6) or [api/Forwards/Create](https://doc.api.afex.com/?version=latest#de38bc9f-dce5-4d58-89ec-e81ff50b7da3) to fix the rate for a trade at the quoted rate (providing the quote has not yet expired).  A quote is valid for 30 seconds.
		
		**Request Parameters**
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Amount|The Trade (buy) amount of the transaction. May be set as Settlement (sell) amount if 'True' is passed in [IsAmountSettlement] parameter.|**True**|
		|CurrencyPair|The currency pair specifies the trade (buy) and settlement (sell) currencies being quoted. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair. For example, if the client wants to convert AUD to USD, the currency pair would be expressed as USDAUD in the quote.|**True**|
		|IsAmountSettlement|Passing "True" will set 'Amount' to settlement (sell) currency.|False|
		|OptionDate|If the quote is for a window / flex forward, the OptionDate must be supplied in the [api/Forwards/Create](https://doc.api.afex.com/?version=latest#de38bc9f-dce5-4d58-89ec-e81ff50b7da3). The option date is the start date of the forward window period.|True *for a window forward only*|
		|ValueDate|The date the proceeds of the trade are paid out (providing the client has remitted settlement funds to AFEX).|**True**|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v2/quote"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'IsAmountSettlement': isamountsettlement, 'OptionDate': optiondate, 'ValueDate': valuedate, 'CurrencyPair': currencypair, 'Amount': amount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencylist(content_type: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details for all AFEX supported currencies.  The details include the ISO currency code required to setup beneficiaries and create trades in the currency.  Data returned on clearing methods per currency i.e. whether HighValue (SWIFT) and / or low value (local clearing) is available for the currency, is used to determine if high or low value clearing may be specified when validating and/or creating a beneficiary in that currency.
		
		**Response Parameters**
		------------
		
		| Parameter  |  Description                            |
		|-----------|-------------------------------------------------------------------|
		|Code      | 3 character ISO currency code e.g. USD, AUD, CHF.              |
		|Name      | Name of currency.<br/> e.g. United States Dollar, Australian Dollar, Swiss Franc.  |
		|HighValue | HighValue indicates this currency will be sent via the SWIFT network.  <br/>SWIFT is same day settlement *providing the payment is submitted within daily processing cut off times*.|
		|LowValue  | LowValue indicates this currency will be send using local clearing.  <br/>Low-value may take up to 3 business days to settle however is lower cost per payment than SWIFT.|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/currency"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findbeneficiary(api_key: str, content_type: str, vendorid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for beneficiary using unique VendorId. Returns a full set of beneficiary data for the Vendor ID supplied as a URI parameter.  Returns standard and instant beneficiaries.
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|VendorId|The unique reference associated with the AFEX beneficiary.  This is set when creating the beneficiary using [api/BeneficiaryCreate](https://doc.api.afex.com/view/256293/S1Lwyo6V?version=latest#b3d5c7d3-c901-4ebb-a916-2ac13f53db1c).|True|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/beneficiary/find"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'VendorId': vendorid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvaluedate(content_type: str, api_key: str, valuetype: str, currencypair: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The value date is the date that the trade proceeds are paid out to either the beneficiary, or in the case of a funding trade, to the client's AFEX currency holding.  Value dates must be business days for both the buy (trade) and sell (settlement) currencies.  The AFEX value date calendar is based on local business dates, business dates in the country of the trade and settlement currency and local payment cut off times by currency. 
		
		
		The api/valuedates method returns next available value date for the ValueType supplied.  ValueType can be CASH (value today), TOM (value next available business day), or SPOT (value 2 business days).  If the value type specified is not allowed for the currency due to AFEX currency rules, an error message will be returned rather than a date.  Value dates beyond spot i.e. forward value dates, are not available via the api/valuedates method.  These can be calculated using inputs from the api/holidays method response.
		
		Value dates returned by the [api/valuedates](https://doc.api.afex.com/?version=latest#f8a2ef3b-0ed0-478f-9a03-9cf5337c9177) call can be used to populate the ValueDate field in the following calls: [api/trades/create](https://doc.api.afex.com/?version=latest#91dda474-6ac2-435c-8999-e2f9302130c6), [api/payments/create](https://doc.api.afex.com/?version=latest#9a8d48e2-012e-4f05-8f43-bb238cf705da), [api/drawdown/create](https://doc.api.afex.com/?version=latest#db41f516-bf89-405c-9cb5-e320ce27fd16), and [api/predelivery/create](https://doc.api.afex.com/?version=latest#5b4dfe12-abe9-4f23-9c07-d3eeabd0debf).
		 
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|CurrencyPair|3-character ISO currency codes for the buy and sell currencies. The currency pair specifies the trade (buy) and settlement (sell) currencies. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair e.g. USDAUD for selling AUD to buy USD.|**True**|
		|ValueType|CASH, TOM or SPOT.  CASH=today, TOM=next business day,  SPOT=2 business days.  If a ValueType is not specified, the SPOT value type will be defaulted.   If the value type is not allowed for the currency provided, an error message will be returned.|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/valuedates"
    querystring = {'Content-Type': content_type, 'API-Key': api_key, 'ValueType': valuetype, 'CurrencyPair': currencypair, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creditlimits(api_key: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns approved trading limits and available limits for cash, tom, spot and forward trades.<br/><br/>*Note, clients approved for forwards are also limited to booking for value dates within their maximum approved forward tenor. Data on a client's approved forward tenor is not available via the API.*   
		
		All limit amounts are expressed in the client's base currency.
		
		| Parameter  |  Description                            |
		|-----------|-------------------------------------------------------------------|
		|SpotLimit      | Client’s approved maximum limit for aggregated unsettled cash, tom and spot transactions.|
		|AvailableSpotLimit      | SpotLimit less total unsettled cash/tom/spot trades booked. Amount available to the client out of the spot limit to book new cash/tom/spot trades.|
		|ForwardLimit | Client’s approved limit for aggregated unsettled outright and window forward transactions.|
		|AvailableForwardLimit  | Forward limit less unsettled forwards booked. Amount available to the client out of the forward limit to book new forward trades.|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/credit"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def identification_types_account(content_type: str, authorization: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of identification types for use with [api/v1/CorporateClient](https://doc.api.afex.com/?version=latest#02991c42-6ae4-45a3-ac54-46e2f77b0a6c) and [api/v1/PrivateClient](https://doc.api.afex.com/?version=latest#993c336c-c9d2-4572-bcce-540bba557530)."
    
    """
    url = f"https://afex2.p.rapidapi.com/api/v1/primaryAccountIdentificationType"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, 'API-Key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getquoteforpredeliveryonoutrightforward(api_key: str, content_type: str, valuedate: str, parenttradenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a rate quote for a predelivery on an existing outright forward.
		
		The quote ID returned may then be supplied when calling api/predelivery/create. A quote is valid for 30 seconds.
		
		**Response Parameters**
		------------
		
		| Parameter  |  Description                            |
		|-----------|-------------------------------------------------------------------|
		|AdjustedInvertedRate      |   The inverted rate quoted for the predelivery trade.|
		|AdjustedRate      | The rate quoted for the predelivery trade.|
		|InvertedRate      |   The inverted rate for the parent trade.|
		|QuoteId      |  The unique AFEX identifier for the quote. The QuoteID can be supplied in api/predelivery/create to fix the rate for a trade providing the trade is created before the quote expires.|  
		|Rate      | The rate for the parent trade.|
		|Terms      |     Terms define whether you need to multiply OR divide amounts by the Rate to calculate the equivalent foreign currency amount.<br/> Converting a Buy Amount to a Sell Amount<br/>If the terms are A, multiply the buy (TradeCcy) amount by the Rate to calculate the sell currency (SettlementCcy) equivalent amount. If the terms are E, divide the buy amount by the Rate to calculate the sell equivalent amount. <br/>Converting a Sell Amount to a Buy Amount<br/>If converting a sell amount to a buy amount, if the terms are A, divide the sell amount by the Rate, if the terms are E, multiply the sell amount by the Rate. <br/>Note, do not use the InvertedRate to calculate a settlement amount. This may result in a rounding discrepancy between the calculated settlement amount and the Settlement Amount provided in the AFEX trade confirmation.|
		|ValueDate      |   The date the proceeds of the predelivery are paid out (providing the client has remitted settlement funds to AFEX). <br/>*Note, available value dates for cash, tom and spot are returned by api/valuedates.*|"
    valuedate: Required.  YYYY/MM/DD. The date the trade currency funds will be paid out.   For a predelivery, the value date cannot be greater than 14 days from the date of creation of the predelivery (inclusive of weekends and public holidays).
        parenttradenumber: Required. TradeNumber (AFEXtransaction ID) of parent forward
        
    """
    url = f"https://afex2.p.rapidapi.com/api/predelivery/quote"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'ValueDate': valuedate, 'ParentTradeNumber': parenttradenumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrate(api_key: str, content_type: str, currencypair: str, valuetype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a rate for a specified currency pair.
		
		This is an indicative rate only.  Use api/quote for securing a quote and booking a transaction.<br/>
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|CurrencyPair|Required.  ISO 3 character currency code for each currency in the pair e.g.  USDAUD.  The currency pair specifies the trade (buy) and settlement (sell) currencies. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair. |**True**|
		|ValueType|Optional. Specify CASH, TOM or SPOT. The default is SPOT if not specified.|False|
		
		**Response Parameters**
		------------
		
		| Parameter  |  Description                            |
		|-----------|-------------------------------------------------------------------|
		|CurrencyPair      | The currency pair specifies the trade (buy) and settlement (sell) currencies. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair.<br/> For example, if the client wants to convert AUD to USD (I.E. sell AUD and buy USD) the currency pair would be expressed as USDAUD.|
		|InvertedRate      | The indicative inverted rate for the currency trade.|
		|Rate      | The indicative rate for the currency trade.|
		|Terms      | Terms define whether you need to multiply OR divide amounts by the Rate to calculate the equivalent foreign currency amount.<br/>Converting a Buy Amount to a Sell Amount<br/>If the terms are A, multiply the buy (TradeCcy) amount by the Rate to calculate the sell currency (SettlementCcy) equivalent amount. If the terms are E, divide the buy amount by the Rate to calculate the sell equivalent amount. If using the InvertedRate to the reverse logic applies.<br/>  Converting a Sell Amount to a Buy Amount<br/>  If converting a sell amount to a buy amount, if the terms are A, divide by the Rate, if the terms are E, multiply by the Rate. If using the InvertedRate, the inverse logic applies.|
		|ValueType      | Cash=value today, Tom=value next available business day, Spot=value two business days.<br/>  If a ValueType is not specified, Spot ValueType will be defaulted.  |"
    currencypair: Required.  ISO 3 character currency code for each currency in the pair e.g.  USDAUD.  The currency pair specifies the trade (buy) and settlement (sell) currencies. The pair must have the trade currency (buy) as the first in the pair and the settlement currency (sell) as the second in the pair. 
        valuetype: Optional. Specify CASH, TOM or SPOT. The default is SPOT if not specified.
        
    """
    url = f"https://afex2.p.rapidapi.com/api/rates"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'CurrencyPair': currencypair, 'ValueType': valuetype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencybalances(api_key: str, content_type: str, enddate: str, startdate: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get balances for each foreign currency holding (Vostro) a client has with AFEX.  If a currency is supplied as a URI parameter, a **funding balance detail report** including all debits and credits for the currency holding within the **current month will be returned.**  Alternatively a date range within the last 120 days can be supplied to return transactions within that time period.
		
		*Note: A Vostro holding / funding balance is automatically created when a client first trades in a currency - there is no API call or manual process required to create a Vostro.*
		
		**Request Parameters**
		------------
		
		|Parameter	|Description	|Required	|
		|-----------|-----------|-----------|
		|Currency|3-character ISO currency code. If a currency is supplied, a transaction list or funding balance detail report for the currency holding (Vostro) will be returned. This method returns debit and credit transactions within the current month i.e. the first of the current month to today’s date unless a date range is specified.|False|
		|StartDate|YYYY/MM/DD. Date range available = [Current Day - 120 days].|False|
		|EndDate|YYYY/MM/DD. Date range available = [Current Day - 120 days].|False|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/funding"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'EndDate': enddate, 'StartDate': startdate, 'Currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def settlementaccountdetails(api_key: str, content_type: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get SSI (standard settlement instructions) for the sell currency.  These are the AFEX bank account details to enable the client to wire the settlement (sell) currency amount to AFEX to fund the trade.  The AFEX bank account varies depending on the sell currency.
		
		**Response Parameters**
		------------
		
		| Parameter  |  Description                            |
		|-----------|-------------------------------------------------------------------|
		|Currency      | The settlement (sell) currency.|
		|PaymentInstructions      | AFEX bank account details and other remittance details required for the client to wire the settlement currency funds to AFEX. |"
    currency: Required.  3 character ISO currency code
        
    """
    url = f"https://afex2.p.rapidapi.com/api/ssi/GetSSI"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validatebeneficiaryiban(api_key: str, content_type: str, country: str, iban: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate the beneficiary's IBAN and return SwiftBIC. The IBAN is the standard international bank account number format used by European banks and is also used in some non-European countries.  The bank SwiftBIC can be derived from the IBAN.
		
		For use when calling the following methods:
		
		1. [api/BeneficiaryCreate](https://doc.api.afex.com/?version=latest#dce44b8e-5798-472c-80ac-161aaf2f70d7)
		2. [api/BeneficiaryUpdate](https://doc.api.afex.com/?version=latest#e3a7d23e-f7b2-4789-873a-b6ccf38e6b15)
		
		**Request Parameters**
		------------
		
		|Parameter|Description|Required|
		|---------|-----------|--------|
		|Iban|Beneficiary bank IBAM|True|
		|Country|Codes are available via [api/Country/Find](https://doc.api.afex.com/?version=latest#36bf62c1-96c8-4e1d-a51c-13fab3ace7b5)|**True**|"
    
    """
    url = f"https://afex2.p.rapidapi.com/api/beneficiary/isiban"
    querystring = {'API-Key': api_key, 'Content-Type': content_type, 'Country': country, 'Iban': iban, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afex2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

