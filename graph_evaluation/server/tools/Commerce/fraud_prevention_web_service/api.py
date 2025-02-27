import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fraudlabs_pro_screen_order(ip: str, format: str, bill_city: str=None, bill_state: str=None, bill_country: str=None, bill_zip_code: str=None, ship_addr: str=None, ship_city: str=None, ship_state: str=None, ship_country: str=None, ship_zip_code: str=None, email_domain: str=None, user_phone: str=None, email_hash: str=None, username_hash: str=None, password_hash: str=None, bin_no: str=None, bin_bank_name: str=None, bin_bank_phone: str=None, bin_bank_country: str=None, card_hash: str=None, avs_result: str=None, cvv_result: str=None, user_order_id: str=None, user_order_memo: str=None, amount: int=None, quantity: int=None, currency: str=None, department: str=None, payment_mode: str=None, session_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    ip: IP address of online transaction. It supports both IPv4 and IPv6 address format.
        format: Return the result in json or xml format.
        bill_city: City of billing address.
        bill_state: State of billing address.
        bill_country: Country of billing address.
        bill_zip_code: Postal or ZIP code of billing address.
        ship_addr: Street address of shipping address.
        ship_city: City of shipping address.
        ship_state: State of shipping address.
        ship_country: Country of shipping address.
        ship_zip_code: Postal or ZIP code of shipping address.
        email_domain: Domain name of email address. For example, the domain of email address support@fraudlabspro.com is fraudlabspro.com.
        user_phone: User's phone number.
        email_hash: SHA1-64k hash of user's email address. No plaintext information. Please visit http://www.fraudlabspro.com/developer/api/screen-order for details.
        username_hash: SHA1-64k hash of user's email address. No plaintext information. Please visit http://www.fraudlabspro.com/developer/api/screen-order for details.
        password_hash: SHA1-64k hash of user's email address. No plaintext information. Please visit http://www.fraudlabspro.com/developer/api/screen-order for details.
        bin_no: First 6 digits of credit card number to identify issuing bank.
        bin_bank_name: Name of the bank which issued the credit card.
        bin_bank_phone: Customer service phone number listed on the back of credit card.
        bin_bank_country: Country of the bank which issued the credit card. It requires the input of ISO-3166 alpha-2 country code, e.g. US for United States.
        card_hash: SHA1-64k hash of user's email address. No plaintext information. Please visit http://www.fraudlabspro.com/developer/api/screen-order for details.
        avs_result: The single character AVS result returned by the credit card processor.
        cvv_result: The single charcater CVV2 result returned by the credit card processor.
        user_order_id: Merchant identifier to uniquely identify a transaction. It supports maximum of 15 characters user order id input.
        user_order_memo: Merchant description of an order transaction. It supports maximum of 200 characters.
        amount: Amount of the transaction.
        quantity: Total quantity of the transaction.
        currency: Currency code used in the transaction. It requires the input of ISO-4217 (3 characters) currency code, e.g. USD for US Dollar.
        department: Merchant identifier to uniquely identify a product or service department.
        payment_mode: Payment mode of transaction. Valid values: creditcard | paypal | googlecheckout | cod | moneyorder | wired | bankdeposit | others.
        session_id: Session ID of server script.
        
    """
    url = f"https://chrislim-fraudlabs-pro-screen-order.p.rapidapi.com/order/screen"
    querystring = {'ip': ip, 'format': format, }
    if bill_city:
        querystring['bill_city'] = bill_city
    if bill_state:
        querystring['bill_state'] = bill_state
    if bill_country:
        querystring['bill_country'] = bill_country
    if bill_zip_code:
        querystring['bill_zip_code'] = bill_zip_code
    if ship_addr:
        querystring['ship_addr'] = ship_addr
    if ship_city:
        querystring['ship_city'] = ship_city
    if ship_state:
        querystring['ship_state'] = ship_state
    if ship_country:
        querystring['ship_country'] = ship_country
    if ship_zip_code:
        querystring['ship_zip_code'] = ship_zip_code
    if email_domain:
        querystring['email_domain'] = email_domain
    if user_phone:
        querystring['user_phone'] = user_phone
    if email_hash:
        querystring['email_hash'] = email_hash
    if username_hash:
        querystring['username_hash'] = username_hash
    if password_hash:
        querystring['password_hash'] = password_hash
    if bin_no:
        querystring['bin_no'] = bin_no
    if bin_bank_name:
        querystring['bin_bank_name'] = bin_bank_name
    if bin_bank_phone:
        querystring['bin_bank_phone'] = bin_bank_phone
    if bin_bank_country:
        querystring['bin_bank_country'] = bin_bank_country
    if card_hash:
        querystring['card_hash'] = card_hash
    if avs_result:
        querystring['avs_result'] = avs_result
    if cvv_result:
        querystring['cvv_result'] = cvv_result
    if user_order_id:
        querystring['user_order_id'] = user_order_id
    if user_order_memo:
        querystring['user_order_memo'] = user_order_memo
    if amount:
        querystring['amount'] = amount
    if quantity:
        querystring['quantity'] = quantity
    if currency:
        querystring['currency'] = currency
    if department:
        querystring['department'] = department
    if payment_mode:
        querystring['payment_mode'] = payment_mode
    if session_id:
        querystring['session_id'] = session_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrislim-fraudlabs-pro-screen-order.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

