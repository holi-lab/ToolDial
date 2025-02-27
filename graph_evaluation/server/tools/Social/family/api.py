import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def register_注册(ac: str, registersubmit: str, accede: str, username: str, password: str, seccode: str, seccode_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "用户注册"
    ac: 操作参数
        registersubmit: 固定参数
        accede: 固定参数
        username: 用户名
        password: 密码
        seccode: 验证码
        seccode_auth: 验证码key，由获取验证码接口返回
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/do.php"
    querystring = {'ac': ac, 'registersubmit': registersubmit, 'accede': accede, 'username': username, 'password': password, 'seccode': seccode, 'seccode_auth': seccode_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def changename_更改昵称(name: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "更改昵称"
    name: 用户名
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'name': name, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publishblog_发布日记(makefeed: str, friend: str, tags: str, subject: str, message: str, friends_1: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "发布日记"
    makefeed: 1代表产生动态,0代表不产生动态
        friend: 隐私，0代表全站可见，1代表家人可见，2代表自己可见
        tags: 标签
        subject: 主题，这里是必须的。注意与发布图片区分
        message: 包含的图片和信息，图片链接由上传图片返回
        friends_1: 中括号里的下标为用户id, familyday是用户名，可以关联多个用户
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'makefeed': makefeed, 'friend': friend, 'tags': tags, 'subject': subject, 'message': message, 'friends[1]': friends_1, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseccode_获取验证码(op: str, mobile: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "获取验证码"
    op: 操作类型
        mobile: 注册的手机号
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/do.php"
    querystring = {'op': op, 'mobile': mobile, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publishpm_发布私信(username: str, message: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "发布私信"
    username: 接收私信的用户名
        message: 私信内容
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'username': username, 'message': message, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rephoto_转载图片(makefeed: str, m_auth: str, photoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "转载图片"
    makefeed: 1代表产生动态，0代表不产生动态
        m_auth: 由登陆或注册成功后返回的用户验证
        photoid: 转载的图片id
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'makefeed': makefeed, 'm_auth': m_auth, 'photoid': photoid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pmdetail_私信详情(pmid: str, touid: str, daterange: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "私信详情"
    pmid: 私信id
        touid: 消息送至的用户id
        daterange: 检索消息的区间（几天之内的）
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/space.php"
    querystring = {'pmid': pmid, 'touid': touid, 'daterange': daterange, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addfrined2_通过家人请求(uid: str, gid: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "通过家人请求"
    uid: 发送家人请求的用户id
        gid: 好友组，0代表全站，1代表家人，2代表其它
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'uid': uid, 'gid': gid, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def friendfeed_好友动态列表(page: str, uid: str, idtype: str='photoid', keyword: str='孩子', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "好友动态列表"
    page: 分页的当前页号
        uid: 用户id
        idtype: feed类别，若略去代表返回所有feed，photoid代表图片，blogid代表日记 ， videoid代表视频，eventid代表活动
        keyword: 搜索参数，一般略去，则返回所有结果
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/do.php"
    querystring = {'page': page, 'uid': uid, }
    if idtype:
        querystring['idtype'] = idtype
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfriends_获取家人列表(uid: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "获取家人列表"
    uid: 用户id
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/space.php"
    querystring = {'uid': uid, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def login_登陆(username: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "用户登陆"
    username: 用户名
        password: 用户密码
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/do.php"
    querystring = {'username': username, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allfeed_全站动态列表(page: str, uid: str, idtype: str='photoid', keyword: str='孩子', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "全站动态列表"
    page: 分页的当前页号
        uid: 用户id
        idtype: feed类别，若略去代表返回所有feed，photoid代表图片，blogid代表日记 ， videoid代表视频，eventid代表活动
        keyword: 搜索参数，一般略去，则返回所有结果
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/do.php"
    querystring = {'page': page, 'uid': uid, }
    if idtype:
        querystring['idtype'] = idtype
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reblog_转载日记(makefeed: str, blogid: str, m_auth: str, message: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "转载日记"
    makefeed: 1代表产生动态,0代表不产生动态
        blogid: 转载的日记 id
        m_auth: 由登陆或注册成功后返回的用户验证
        message: 这个参数应该是一个bug，待修正
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'makefeed': makefeed, 'blogid': blogid, 'm_auth': m_auth, 'message': message, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pm_私信列表(page: str, prepage: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "私信列表"
    page: 当前页号
        prepage: 每页列表数
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/space.php"
    querystring = {'page': page, 'prepage': prepage, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def doing_编写心情(message: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "编写心情"
    message: 心情内容
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'message': message, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def notice_通知列表(page: str, prepage: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "通知列表"
    page: 当前页号
        prepage: 每一页条数
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/space.php"
    querystring = {'page': page, 'prepage': prepage, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addfriend_单向请求加家人(uid: str, gid: str, note: str, m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "单向请求加家人"
    uid: 请求加为好友的用户id
        gid: 好友组id号，0全站，1家人，2其他
        note: 请求消息
        m_auth: 由登陆或注册成功后返回的用户验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'uid': uid, 'gid': gid, 'note': note, 'm_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_撰写评论(m_auth: str, idtype: str, is_id: str, message: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "撰写评论"
    m_auth: 由登陆或注册成功后返回的用户验证
        idtype: 取值范围：photoid代表评论图片，blogid代表评论日记
        id: 评论文章的id号
        message: 评论内容
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'m_auth': m_auth, 'idtype': idtype, 'id': is_id, 'message': message, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publishphoto_发布图片(makefeed: str, title_2000: str, tags: str, friend: str, m_auth: str, subject: str='孩子', friends_1: str='familyday', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "发布图片"
    makefeed: 1代表产生动态,0代表不产生动态
        title_2000: title中括号里面的是图片id，由上传图片返回，值是图片的描述
        tags: 标签，用于标识图片所属相册
        friend: 隐私，0代表全站可见，1代表家人可见，2代表自己可见
        m_auth: 由登陆或注册成功后返回的用户验证
        subject: 图片标题
        friends_1: 中括号里的下标为用户id, familyday是用户名，可以关联多个用户
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'makefeed': makefeed, 'title[2000]': title_2000, 'tags': tags, 'friend': friend, 'm_auth': m_auth, }
    if subject:
        querystring['subject'] = subject
    if friends_1:
        querystring['friends[1]'] = friends_1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def logout_退出(m_auth: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "退出"
    m_auth: 身份验证
        
    """
    url = f"https://summit4you-family.p.rapidapi.com/cp.php"
    querystring = {'m_auth': m_auth, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summit4you-family.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

