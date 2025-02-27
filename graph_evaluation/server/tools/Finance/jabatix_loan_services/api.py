import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrievalofriskratiosforloansviaid_companyandgaapatspecificpostingdate(dealid: str, postingdate: str, companyid: int, gaap: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>
		      The output interface &#8222;GET loans/risk-ratios&#8220; provides the most important 
		      risk ratios.
		    </p>
		    <p>
		      <br>
		      Detailed information can be found here: <a href="https://confluence.jabatix.net/x/fAlbAw" target=_blank>GET 
		      /loans/risk-ratios</a>
		    </p>"
    dealid: Unique deal id
        postingdate: (Required) Posting date for which data is requested
        companyid: (Required) Unique company id
        gaap: Accounting guideline
        
    """
    url = f"https://jabatix-loan-services.p.rapidapi.com/risk-ratios"
    querystring = {'dealId': dealid, 'postingDate': postingdate, 'companyId': companyid, 'gaap': gaap, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jabatix-loan-services.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievalofaccountingtransactionsandrelatedratiosforloansviaid_companyandgaapatspecificpostingdate(dealid: str, companyid: int, gaap: str, postingdate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>
		      The output interface &quot;Accounting Transactions and related Accounting 
		      Ratios&quot; provides for each individual deal for various accounting 
		      transactions values for accounting ratios. These accounting ratios form 
		      the basis for the generation of debit/credit entries.
		    </p>
		    <p>
		      An accounting transaction is defined by the combination of
		    </p>
		    <ul>
		      <li>
		        an accounting event type, such as payment or periodical valuation 
		        requests and
		      </li>
		      <li>
		        an accounting event.
		      </li>
		    </ul>
		    <p>
		      <br>
		      Detailed information can be found here: <a href="https://confluence.jabatix.net/x/agdbAw" target=_blank>GET 
		      loans/accounting-transaction-related-accounting-ratios</a>
		    </p>"
    dealid: Unique deal id
        companyid: (Required) Unique company id
        gaap: Accounting guideline
        postingdate: (Required) Posting date for which data is requested
        
    """
    url = f"https://jabatix-loan-services.p.rapidapi.com/accounting-transaction-related-accounting-ratios"
    querystring = {'dealId': dealid, 'companyId': companyid, 'gaap': gaap, 'postingDate': postingdate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jabatix-loan-services.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievalofdebitcreditentriesforloansviaid_companyandgaapatspecificpostingdate(gaap: str, dealid: str, postingdate: str, companyid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>
		      The output interface &#8222;GET Debit/Credit Entries&#8220; provides for each 
		      individual deal and related accounting transaction the appropriate set 
		      of entries. For each set of entry debit and credit bookings are 
		      provided, taking a standard chart of accounts into account.
		    </p>
		    <p>
		      Set of entries provided cover the full life cycle of a loan. Different 
		      accounting event types such as contractual maturity, payment, accrual 
		      and fair valuing ensure timely generation.&#160; For each accounting event 
		      type relevant accounting events are considered, appropriate set of 
		      entries identified and related debit/credit entries on a standard chart 
		      of accounts generated.
		    </p>
		    <p>
		      <br>
		      Detailed information can be found here: <a href="https://confluence.jabatix.net/x/XgdbAw" target=_blank>GET 
		      /loans/debit-credit-entries</a>
		    </p>"
    gaap: Accounting guideline
        dealid: Unique deal id
        postingdate: (Required) Posting date for which data is requested
        companyid: (Required) Unique company id
        
    """
    url = f"https://jabatix-loan-services.p.rapidapi.com/debit-credit-entries"
    querystring = {'gaap': gaap, 'dealId': dealid, 'postingDate': postingdate, 'companyId': companyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jabatix-loan-services.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievalofgrossbookvalueevidenceforloansviaid_companyandgaapatspecificpostingdate(companyid: int, gaap: str, postingdate: str, dealid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>
		      The output interface &#8220;Gross Book Value Evidence&#8221; provides valuation 
		      elements which are part of the gross book value on individual deal level.<br>
		    </p>
		    <p>
		      <br>
		      Detailed information can be found here: <a href="https://confluence.jabatix.net/x/WAdbAw" target=_blank>GET 
		      /loans/gross-book-value-evidence</a>
		    </p>
		    <br>"
    companyid: (Required) Unique company id
        gaap: Accounting guideline
        postingdate: (Required) Posting date for which data is requested
        dealid: Unique deal id
        
    """
    url = f"https://jabatix-loan-services.p.rapidapi.com/gross-book-value-evidence"
    querystring = {'companyId': companyid, 'gaap': gaap, 'postingDate': postingdate, 'dealId': dealid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jabatix-loan-services.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

