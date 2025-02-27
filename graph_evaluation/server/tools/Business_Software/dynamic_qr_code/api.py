import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dynamic_qr_code_api(qrtext: str, output: str='json', back: str='6C3483', size: int=250, website: str='www.softsysinfo.com', front: str='922B21', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<div class="row"> 
		 <div class="col-sm-12">
		<div class="card mb-2 mt-3  maindiv">
		  <div class="card-header">         
		        <h2 class="text-center">Dynamic QR Code API- Developers Resource</h2> 
		     <p>Developers can integrate our QR Code API ( Application Programming Interface) into their web applications or mobile applications.  <br>
		Generate Dynamic QR Codes for Targeted Marketing Compaigns and Lead Generations</p>
		<h3>API END POINT: https://www.omqrc.com/qrcodeapi</h3>
		  </div>
		 <div class="card-body">      
		       
		 <div class="row"> 
		 <div class="col-sm-12 mr-2 ml-2 border">
		<div class="table-responsive">
		 <table class="table table-bordered table-striped" ><thead><tr><th>Parameters</th><th>Data Type</th><th>Whether Mandatory</th><th>Description</th><tr></thead>
		 <tbody>
		 <tr>
		<tr><td>apikey</td><td>	string</td><td>	Yes	</td><td>Register and get API Key from member area</td></tr>
		<tr><td>seckey</td><td>	string</td><td>	Yes	</td><td>Login to member area and set Secret Key</td></tr>
		<tr><td>qrtext</td><td>	string</td><td>	Yes	</td><td>Information to be encoded in QR Code. Website URL in case you want to encode in QR Code</td></tr>
		 <tr><td>size	</td><td>Number  between 150-1000</td><td>	No</td><td>	Size of the UPI QR Code to be generated. If blank, then default size will be applied</td></tr>
		<tr><td>front</td><td>	Hexa Color Code for foreground</td><td>	No	</td><td>If not supplied , then black and white UPI QR Code with be generated.</td></tr>
		<tr><td>back</td><td>	Hexa Color Code for background</td><td>	No</td><td>	If not supplied , then black and white UPI QR Code with be generated.</td></tr>
		<tr><td>logo</td><td>	Logo file URL	</td><td>No	</td><td>If Logo is not supplied,  UPI QR Code till be generated  without Logo. You should check whether QR Code is readable after displaying logo.</td></tr>    
		 
		</tbody>
		</table>
		</div>
		</div>
		</div>
		</div>
		</div>
		</div>"
    qrtext: You can send text or url in the parameter
        output: QR Code outout parameter. In case you require Json data in response, then define json in parameter. If the parameter is not defined, then you will get image in the output. 
        back: Background Color of QR Code Image. 6 character hexa color code without #, i.e. 6C3483.  By default white background color will be used
        size: Size of the QR Code to be generated. You can define 50-1000 in the parameter value. By default 500px QR Code will be generated
        website: Your website as www.upiqrcode.com
        front: Foreground color of the QR Code to be used, 6 character hexa color code without #, i.e.  922B21 .  By default black color be used
        
    """
    url = f"https://dynamic-qr-code1.p.rapidapi.com/qrcodeapi"
    querystring = {'qrtext': qrtext, }
    if output:
        querystring['output'] = output
    if back:
        querystring['back'] = back
    if size:
        querystring['size'] = size
    if website:
        querystring['website'] = website
    if front:
        querystring['front'] = front
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dynamic-qr-code1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

