import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cryptocrit(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# CryptoCRIT
		
		![CryptoCRIT](https://theabbie.github.io/files/CryptoCRIT.png)
		
		## Introduction
		
		CryptoCRIT is an open-source Cryptocurrency project which facilitates a Cryptocurrency wallet for making payments. The associated Cryptocurrency is called CritCoin.
		
		This system will be used in [Fr. Conceicao Rodrigues Institute of Technology](https://www.fcrit.ac.in)
		
		<table>
		<caption>CryptoCRIT</caption>
		<thead>
		<tr>
		<th colspan="2">Quick Info</th>
		</tr>
		</thead>
		<tbody>
		<tr><th scope='row'>Name</th><td>CryptoCRIT</td></tr>
		<tr><th scope='row'>Technologies Used</th><td><ul><li>Blockchain</li><li>RSA Encryption</li></ul></td></tr>
		<tr><th scope='row'>College</th><td><a href="https://www.fcrit.ac.in">Fr. Conceicao Rodrigues Institute of Technology</a></td></tr>
		<tr><th scope='row'>Team</th>
		<td>
		<ul>
		<li><a href="https://github.com/rgab1508">Gabriel Rajendran</a></li>
		<li><a href="https://instagram.com/adi.unni">Aditya Unnikrishnan</a></li>
		<li><a href="https://theabbie.github.io">Abhishek Chaudhary</a></li>
		<li><a href="https://instagram.com/tushar_gupta_339">Tushar Gupta</a></li>
		</ul>
		</td>
		</tr>
		</tbody>
		</table>
		
		## Tasks
		
		- [ ] Backend
		- [ ] Frontend
		- [ ] Logic
		- [ ] Working
		- [ ] Production
		
		## Usage
		
		```sh
		mkdir <project-name>
		cd <project-name>
		cryptocrit run
		```
		
		## Contributing
		
		Thank you for your interest in contributing, We are currently not interested in Contribution as this is our college project, If you have any suggestions or ideas, you can contact any of us, details are as mentioned below
		
		## Help
		
		You can contact us on social media, Our preferred mediums are as mentioned below
		
		## Installation
		
		### Requirements
		
		* Node.Js installed
		* A server or localhost
		
		### Installation
		
		A sample installation would be something like this
		
		```sh
		npm install -g cryptocrit
		cryptocrit init
		```
		
		### Configuration
		
		All the configuration data should be present in a `config.json` file
		
		## Credits
		
		* [Gabriel Rajendran](https://github.com/rgab1508)
		* [Aditya Unnikrishnan](https://instagram.com/adi.unni)
		* [Abhishek Chaudhary](https://theabbie.github.io)
		* [Tushar Gupta](https://instagram.com/tushar_gupta_339)
		
		## Contact
		
		Contact us on [Gitter](https://gitter.im/CryptoCRIT/community)
		
		## License
		
		This project is licensed under MIT License, See [LICENSE](/LICENSE) for more information"
    
    """
    url = f"https://cryptocrit1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocrit1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

