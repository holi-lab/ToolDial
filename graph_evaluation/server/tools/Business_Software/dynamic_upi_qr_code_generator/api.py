import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def https_www_upiqr_com(vpa: str, payee: str, output: str='json', paymode: str='bankac', amount: int=100, billno: str='123456', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API generate Dynamic  UPI QR Code in the form of PNG image or Base64 image format, depending the output parameter, defined. You can display the image on POS, vending machines or print the image on the bills, so payer can scan the UPI QR Code with UPI APPs to make payment."
    vpa: BankAccount@IFSCCode, if receiving payment with bank account & IFSC Code, otherwise upiid@upihandle
        payee: Receiver of the Payment whose vpa is given.

Note: Replace blank Spaces in the Parameter value with %20
        paymode: Parameter is required to receive payment with Bank Account No. & IFSC Code. In case you have UPI ID / VPA, then the parameter is not required.
        billno: Bill Number / Order Number / Transaction Referance
        
    """
    url = f"https://dynamic-upi-qr-code-generator.p.rapidapi.com/dynaupiqrapi"
    querystring = {'vpa': vpa, 'payee': payee, }
    if output:
        querystring['output'] = output
    if paymode:
        querystring['paymode'] = paymode
    if amount:
        querystring['amount'] = amount
    if billno:
        querystring['billno'] = billno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dynamic-upi-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

