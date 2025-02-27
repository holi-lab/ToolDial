import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def https_www_upiqr_com(sgstamount: str, billdate: str, payee: str, cess: str, sellergstin: str, bankacno: str, ifsccode: str, igstamount: str, cgstamount: str, billno: str, mc: str='0000', size: str='200', vpa: str='upiid@upihandle', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate dynamic  B2C QR Codes for GST Invoices.  Print B2C  QR Codes on GST Invoices so your customers can scan with UPI APPs to make you payment and view invoice details encoded in the Invoices."
    
    """
    url = f"https://b2c-qr-code.p.rapidapi.com/gstb2c"
    querystring = {'SGSTAmount': sgstamount, 'billDate': billdate, 'payee': payee, 'CESS': cess, 'SellerGstin': sellergstin, 'BankAcNo': bankacno, 'IFSCCode': ifsccode, 'IGSTAmount': igstamount, 'CGSTAmount': cgstamount, 'billno': billno, }
    if mc:
        querystring['mc'] = mc
    if size:
        querystring['size'] = size
    if vpa:
        querystring['vpa'] = vpa
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "b2c-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

