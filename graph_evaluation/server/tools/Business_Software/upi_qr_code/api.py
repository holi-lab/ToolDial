import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def upi_qr_code_generator(amount: str, payee: str, billno: str, vpa: str, paymode: str='bankac', bgcolor: str='FFFFFF', msid: str='store0001', url: str='https://www.upiqrcode.com', mc: str='mc10001', mtid: str='terminal0001', orgid: str='org00001234', mid: str='m0001', minamt: str='1000.00', frcolor: str='000000', sign: str='merchantsign', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate dynamic UPI QR Codes to print on bills, so your customers can scan the UPI QR Codes with UPI APPs to make you payment."
    payee: Merchant Code Allotted by PSP
        vpa: If you have UPI ID the pass UPI ID
for bank account & IFSC Code pass bankac@ifsccode
        paymode: Parameter is conditional, No need of the parameter if you have UPI ID.
For receiving payment for bank account & IFSC Code pass value bankac
        msid: Merchant Terminal ID
        url: Originating ID
        mtid: Sign
        orgid: Merchant ID
        mid: Merchant Store ID
        minamt: Transaction Reference URL 
        
    """
    url = f"https://upi-qr-code.p.rapidapi.com/generator"
    querystring = {'amount': amount, 'payee': payee, 'billno': billno, 'vpa': vpa, }
    if paymode:
        querystring['paymode'] = paymode
    if bgcolor:
        querystring['bgcolor'] = bgcolor
    if msid:
        querystring['msid'] = msid
    if url:
        querystring['url'] = url
    if mc:
        querystring['mc'] = mc
    if mtid:
        querystring['mtid'] = mtid
    if orgid:
        querystring['orgid'] = orgid
    if mid:
        querystring['mid'] = mid
    if minamt:
        querystring['minamt'] = minamt
    if frcolor:
        querystring['frcolor'] = frcolor
    if sign:
        querystring['sign'] = sign
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upi-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

