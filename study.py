# from random import randint
#
# import pandas as pd
# from collections import OrderedDict
# def make_car(manufacture,model,**others):
#     car={}
#     car['manufacture']=manufacture
#     car['model']=model
#     for k,v in others.items():
#         car[k]=v
#     return car
#
# car = make_car('subaru', 'outback', color='blue', tow_package=True,name="lovoeu")
# print(car)
#
# favorite_l=OrderedDict()
# favorite_l["1"]=1
# favorite_l["2"]=2
# favorite_l["3"]=3
#
# for k,v in favorite_l.items():
#     print(str(k)+" "+str(v))
# print(type(favorite_l))
#
# for i in range(0,10):
#     x=randint(1,10)
#     print(x)

import json
def get_stored_username():
    filename = 'username.txt'
    user = []
    try:
        with open(filename, encoding = 'utf-8') as f_obj:
            for line in f_obj:
                user.append(line.rstrip('\n'))
    except FileNotFoundError:
        return None
    else:
        print('type: {}'.format(type(user)))
        print('all usernames: {}'.format(user) )
        return user

def get_new_username(username):
    if username is None:
        username = input("What is your name? ")
    filename = 'username.txt'
    with open(filename, 'a') as f_obj:
        f_obj.write(username+'\n')
    return username


def greet_user():
    usernames = get_stored_username()
    u = input("please input your name: ")
    for username in usernames:
        if u == username:
            print("Welcome back, " + username + "!")
            return
    print("sorry,you are the new one")
    username = get_new_username(u)
    print("We'll remember you when you come back, " + username + "!")

greet_user()
