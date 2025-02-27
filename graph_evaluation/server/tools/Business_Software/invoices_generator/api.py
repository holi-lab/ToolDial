import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_invoice(services: str, tax: int, currency: str, date: str, buyer_company_name: str, number: str, seller_company_name: str, locale: str='en', buyer_address: str='Buyer Address', seller_tax_number: str='Seller Tax Number', seller_bank_account: str='Seller Bank Account', seller_vat_number: str='Seller VAT Number', logo: str='https://cdn.logo.com/hotlink-ok/logo-social.png', service_fee: int=10, due_date: str='2022-01-01', buyer_vat_number: str='Buyer VAT Number', buyer_tax_number: str='Buyer Tax Number', seller_address: str='Seller Address', shipping: int=30, seller_bank_name: str='Seller Bank Name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quickly generate invoices through a REST API endpoint"
    services: A JSON representing the services, and their prices that will be written on the invoice. The JSON is using the following format:

`[{\\\\\\\"name\\\\\\\": \\\\\\\"My Service\\\\\\\", \\\\\\\"price\\\\\\\": \\\\\\\"30\\\\\\\", \\\\\\\"units\\\\\\\": \\\\\\\"Hours\\\\\\\", \\\\\\\"discount\\\\\\\": \\\\\\\"1000\\\\\\\", \\\\\\\"quantity\\\\\\\": \\\\\\\"1000\\\\\\\"}]`

The JSON is an array. Each element of the array represents a service. Each service is a JSON object with the following properties:

- name: The name of the service/product that is sold
- price: Represents the price per unit of the service/product that is sold
- units: Represents how the service/product in measured in quantity. This can be anything, such as hours, kilograms, tons etc.
-discount: Represents the total discount for all the quantity of this product/service.
-quantity: Represents the amount of units that are being sold. For example if the unit is hour and the quantity is 100, then you're billing 100 hours of the service. Another example could be where the units is kilograms and the quantity is 100 and the product sold is \\\\\\\"Sugar\\\\\\\" => billing 100 kilograms of sugar.
        tax: The tax percentage aplied to the invoice.
        currency: The currency in which the prices will be shown on the invoices. It can be any 3 letter currency code. If the code is not recognized then it will be simply displayed directly.
        date: The date of the invoice
        buyer_company_name: The company name of the buyer entity. It can also be just a simple name of a person in case the buyer is not a company.
        number: Represents the invoice number
        seller_company_name: The company name of the seller entity. This can be a person's name as well, in case the seller is not a company.
        locale: The language of the invoice. Accepted languages are:

'en' => 'English',
'nl' => 'Dutch',
'de' => 'German',
'es' => 'Spanish',
'fr' => 'French',
'it' => 'Italian',
'pl' => 'Polish',
'pt_BR' => 'Brazilian Portuguese',
'ro' => 'Romanian',
        buyer_address: The address of the buyer entity. This field is not mandatory.
        seller_tax_number: The tax number of the seller's company. This field is not mandatory.
        seller_bank_account: The bank account of the seller, where the payment will be made. This field is not mandatory.
        seller_vat_number: The VAT number of the seller's company. This field is not mandatory.
        logo: A logo to display on the top left corner of the invoice. This field is not mandatory.
        service_fee: A service fee to add on the invoice. This field is not mandatory.
        due_date: The due date of the invoice. This field is not mandatory.
        buyer_vat_number: The VAT number of the buyer's company. This field is not mandatory.
        buyer_tax_number: The tax number of the buyer's company. This field is not mandatory.
        seller_address: The address of the seller. This field is not mandatory.
        shipping: The amount to be paid for shipping. This field is not mandatory.
        seller_bank_name: The bank name of the account of the seller. This field is not mandatory.
        
    """
    url = f"https://invoices-generator.p.rapidapi.com/generate-invoice"
    querystring = {'services': services, 'tax': tax, 'currency': currency, 'date': date, 'buyer_company_name': buyer_company_name, 'number': number, 'seller_company_name': seller_company_name, }
    if locale:
        querystring['locale'] = locale
    if buyer_address:
        querystring['buyer_address'] = buyer_address
    if seller_tax_number:
        querystring['seller_tax_number'] = seller_tax_number
    if seller_bank_account:
        querystring['seller_bank_account'] = seller_bank_account
    if seller_vat_number:
        querystring['seller_vat_number'] = seller_vat_number
    if logo:
        querystring['logo'] = logo
    if service_fee:
        querystring['service_fee'] = service_fee
    if due_date:
        querystring['due_date'] = due_date
    if buyer_vat_number:
        querystring['buyer_vat_number'] = buyer_vat_number
    if buyer_tax_number:
        querystring['buyer_tax_number'] = buyer_tax_number
    if seller_address:
        querystring['seller_address'] = seller_address
    if shipping:
        querystring['shipping'] = shipping
    if seller_bank_name:
        querystring['seller_bank_name'] = seller_bank_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "invoices-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

