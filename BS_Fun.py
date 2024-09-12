from math import log, sqrt, pi, exp
from scipy.stats import norm
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from numpy import random as nr

def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+sigma**2/2.)*T)/(sigma*sqrt(T))
def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)
def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
def bs_put(S,K,T,r,sigma):
    return K*exp(-r*T)-S+bs_call(S,K,T,r,sigma)

class BS_Options_Pricing_S():
    def __init__(self,S,other_params):
        self.S = S
        self.other_params = other_params

    def make_calls(self):
        self.calls = []
        for s in self.S:
            self.calls.append(bs_call(s,*self.other_params))

    def make_puts(self):
        self.puts=[]
        for s in self.S:
            self.puts.append(bs_put(s,*self.other_params))

class BS_Options_Pricing_K():
    def __init__(self,K,other_params):
        self.S = other_params[0]
        self.K = K
        self.other_params = other_params[1:]

    def make_calls(self):
        self.calls = []
        for k in self.K:
            self.calls.append(bs_call(self.S,k,*self.other_params))

    def make_puts(self):
        self.puts=[]
        for s in self.S:
            self.puts.append(bs_put(self.S,k,*self.other_params))


class BS_Options_Pricing_sigma():
    def __init__(self,S,K,T,r,sigma):
        self.S = S
        self.K = K
        self.T = T
        self.sig = sigma
        self.r = r

    def make_price_per_k(self,si,cp_fun=bs_call):
        result = []
        for k in self.K:
            result.append(cp_fun(self.S,k,self.T,self.r,si))
        return result


    def make_calls(self):
        self.calls = {}
        for si in self.sig:
            self.calls.update({si:self.make_price_per_k(si)})

class BS_Options_Pricing_sigma_with_noise(BS_Options_Pricing_sigma):
    def __init__(self,S,K,T,r,sigma,mu_noise,sigma_noise):
        super().__init__(S,K,T,r,sigma)
        self.mu_noise       =   mu_noise
        self.sigma_noise    =   sigma_noise

    def bs_call_noise(self,S,K,T,r,sigma):
        noise=nr.normal(self.mu_noise,self.sigma_noise)
        return max(bs_call(S,K,T,r,sigma)+noise,0)

    def make_calls(self,seed=0,number_beauty=4):
        self.calls={}
        nr.seed(seed)
        for si in self.sig:
            prices=self.make_price_per_k(si,cp_fun=self.bs_call_noise)
            name=round(si,number_beauty)
            self.calls.update({name:prices})

# class BS_Options_Pricing_kt():
#     def __init__(self,S,K,T,r,sigma):
#         self.S = S
#         self.K = K
#         self.T = T
#         self.other_params = [r,sigma]

#     def make_price_per_t(self,t,cp_fun=bs_call):
#         result = []
#         for k in self.K:
#             result.append(cp_fun(self.S,k,t,*self.other_params))
#         return result


#     def make_calls(self):
#         self.calls = {}
#         for t in self.T:
#             self.calls.update({t:self.make_price_per_t(t)})

#     def make_calls_beautifultitle(self,number_beauty=3):
#         self.calls = {}
#         for t in self.T:
#             self.calls.update({
#                 (str(round(t,number_beauty))+str('C')):
#                 self.make_price_per_t(t)})

#     def make_puts(self):
#         self.puts = {}
#         for t in self.T:
#             self.puts.update({t:self.make_price_per_t(t,bs_put)})

#     def make_everything_beautifultitle(self,number_beauty=3):
#         self.everything=self.calls
#         self.puts = {}
#         for t in self.T:
#             self.puts.update({
#                 (str(round(t,number_beauty))+str('P')):
#                 self.make_price_per_t(t,bs_put)})
#         self.everything.update(self.puts)



