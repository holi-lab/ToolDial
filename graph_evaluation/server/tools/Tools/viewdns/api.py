import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def whois(domain: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays owner/contact information for a domain name. Can also be used to determine if a domain name is registered or not.  This tool is not available to free API key users. Access is restricted to paid API keys only."
    domain: the domain to perform a whois lookup on
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/whois"
    querystring = {'domain': domain, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def port_scanner(host: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web based port scanner will test whether common ports are open on a server. Useful in determining if a specific service (e.g. HTTP) is up or down on a specific server.   Ports scanned are: 21, 22, 23, 25, 80, 110, 139, 143, 445, 1433, 1521, 3306 and 3389"
    host: the host to perform the port scanner on (domain or IP address)
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/portscan"
    querystring = {'host': host, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traceroute(domain: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Determines the series of servers that data traverses from the ViewDNS server to the specified domain name or IP address."
    domain: the domain or IP address to perform a traceroute on
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/traceroute"
    querystring = {'domain': domain, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dns_record_lookup(domain: str, output: str, recordtype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View all configured DNS records (A, MX, CNAME etc.) for a specified domain name."
    domain: the domain name to lookup DNS records for
        output: the output format required ('xml' or 'json')
        recordtype: the type of DNS record you wish to retrieve (default 'ANY')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/dnsrecord"
    querystring = {'domain': domain, 'output': output, }
    if recordtype:
        querystring['recordtype'] = recordtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ping(output: str, host: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test how long a response from remote system takes to reach the ViewDNS server. Useful for detecting latency issues on network connections."
    output: the output format required ('xml' or 'json')
        host: the domain or IP address to perform a ping on
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/ping"
    querystring = {'output': output, 'host': host, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_location_finder(ip: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This tool will display geographic information about a supplied IP address including city, country, latitude, longitude and more."
    ip: the ip address to find the location of
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/iplocation"
    querystring = {'ip': ip, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chinese_firewall_test(domain: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a site is blocked by the Great Firewall of China. This test checks across a number of servers from various locations in mainland China to determine if access to the site provided is possible from behind the Great Firewall of China.   This test checks for symptoms of DNS poisoning, one of the more common methods used by the Chinese government to block access to websites."
    domain: the domain name to test
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/chinesefirewall"
    querystring = {'domain': domain, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spam_database_lookup(ip: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out if your mail server is listed in any spam databases."
    ip: the IP address to test for spam blacklisting
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/spamdblookup"
    querystring = {'ip': ip, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def free_email_lookup(domain: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out if a domain name provides free email addresses. Search is performed on a custom made list of thousands of known free email hosts."
    domain: the domain name to test for free email services
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/freeemail"
    querystring = {'domain': domain, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mac_address_lookup(mac: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This tool will display the name of the company that manufactured a specific network device based on its MAC Address."
    mac: the MAC address to lookup
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/maclookup"
    querystring = {'mac': mac, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def abuse_contact_lookup(domain: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to find the abuse contact address for a domain name. This is where you would send complaints about spam originating from that domain."
    domain: the domain name to find the abuse contact for
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/abuselookup"
    querystring = {'domain': domain, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dns_propagation_checker(domain: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check whether recent changes to DNS entries have propagated to DNS servers all over the world. Useful in troubleshooting DNS issues that appear to be isolated to one geographic region. Provides a status report on DNS propagation globally."
    domain: the domain name to test
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/propagation"
    querystring = {'domain': domain, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iran_firewall_test(siteurl: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test if any website is accessible using the Internet in Iran in real-time."
    siteurl: the URL to test
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/iranfirewall"
    querystring = {'siteurl': siteurl, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_dns_lookup(ip: str, output: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the reverse DNS entry (PTR) for a given IP. This is generally the server or host name."
    ip: the IP address to retrieve the reverse DNS record for
        output: the output format required ('xml' or 'json')
        
    """
    url = f"https://community-viewdns.p.rapidapi.com/reversedns"
    querystring = {'ip': ip, 'output': output, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-viewdns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

