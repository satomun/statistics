#確率分布シミュレーション

import graphviz as graphviz
import math
import numpy as np
import pandas as pd
from scipy.special import beta, gamma
import streamlit as st
import time
import warnings
warnings.simplefilter('ignore')


#アイデア
#回帰分析も入れていく

radio_d = st.sidebar.radio('メニュー',
                          ("確率分布の関係",
                           "ベルヌーイ分布",
                           "二項分布",
                           "ポアソン分布",
                           "指数分布",
                           "正規分布",
                           "ベータ分布",
                           ))

if radio_d == "確率分布の関係":
    G = graphviz.Digraph()
    # G.node("start", shape="circle", color="pink")
    G.edge('ベルヌーイ分布', '二項分布',label="複数回")
    G.edge('ベルヌーイ分布', '幾何分布',label="初めて成功")
    G.edge('二項分布', 'ポアソン分布',label="λ=np, n→∞")
    G.edge('幾何分布', '指数分布',label="連続化")
    G.edge('幾何分布', '負の二項分布',label="連続化")
    G.edge('指数分布', 'ワイブル分布',label="一般化")
    G.edge('指数分布', 'ガンマ分布',label="一般化")
    G.edge('ガンマ分布', 'ベータ分布',label="一般化")
    G.edge('正規分布', 'カイ二乗分布',label="２乗の線形和")
    G.edge('カイ二乗分布', 't分布',label="")
    G.edge('カイ二乗分布', 'F分布',label="")
    st.graphviz_chart(G)

if radio_d == "ベルヌーイ分布":
    st.write("ベルヌーイ分布")
    p = st.number_input(
        'パラメータp',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )

    x = [0,1]
    y = map(lambda k: p**k * (1-p)**(1-k),x)
    df = pd.DataFrame(data=y,index=x,columns=["y"])
    st.latex(r'''確率関数：P(X=k)=p^k (1-p)^{1-k}''')
    st.latex(r'''平均：p　　分散：p(1-p)''')
    st.bar_chart(df,height=0)

if radio_d == "二項分布":
    st.write("二項分布")
    n = st.number_input(
        'パラメータn',
        min_value=1,
        max_value=100,
        value=6,
    )

    p = st.number_input(
        'パラメータp',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )

    x = np.arange(0, n+1, 1)
    y = map(lambda k:math.factorial(n)/(math.factorial(n-k)*math.factorial(k))*p**(k)*(1-p)**(n-k),x)
    df = pd.DataFrame(data=y,index=x,columns=["y"])
    st.latex(r'''確率関数：P(X=k)=_nC_k p^{k}(1-p)^{n-k}''')
    st.latex(r'''平均：np　　分散：np(1-p)''')
    st.line_chart(df)

if radio_d == "ポアソン分布":
    st.write("ポアソン分布")
    l = st.number_input(
        'パラメータλ',
        min_value=0,
        max_value=100,
        value=2,
    )

    x = np.arange(0, 30, 1)
    y = map(lambda k:(l ** k) * np.exp(-l)/math.factorial(k),x)
    df = pd.DataFrame(data=y,index=x,columns=["y"])
    st.latex(r'''確率関数：P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}''')
    st.latex(r'''平均：\lambda　　分散：\lambda''')
    st.line_chart(df)

if radio_d == "指数分布":
    st.write("指数分布")
    l = st.number_input(
        'パラメータλ',
        min_value=0,
        max_value=100,
        value=2,
    )

    x = np.arange(0, 30, 1)
    y = map(lambda k:l * np.exp(-l*k),x)
    df = pd.DataFrame(data=y,index=x,columns=["y"])
    st.latex(r'''確率密度関数：f(x)=\lambda e^{-\lambda x}''')
    st.latex(r'''平均：\frac{1}{\lambda}　　分散：\frac{1}{\lambda^2}''')
    st.line_chart(df)

if radio_d == "正規分布":
    st.write("正規分布")
    col1, col2 = st.beta_columns(2)
    with col1:
        m = st.number_input(
            'パラメータμ',
            min_value=-100,
            max_value=100,
            value=0,
        )
    with col2:   
        v = st.number_input(
            '分散σ^2',
            min_value=1,
            max_value=100,
            value=1,
        )
    x = np.arange(-10, 10, 0.1)
    y = map(lambda k: 1/np.sqrt(2*np.pi*v) * np.exp(-((k-m)**2)/(2*v)),x)
    # y = map(lambda k:k**2,x)
    df = pd.DataFrame(data=y,index=x,columns=["y"])
    st.latex(r'''
            確率密度関数：f(x)=\frac{1}{\sqrt{2\pi }\sigma}\exp \left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
            ''')
    st.latex(r'''平均：μ　　分散：σ^2''')
    st.line_chart(df)

if radio_d == "ベータ分布":
    st.write("ベータ分布")
    col1, col2 = st.beta_columns(2)
    with col1:
        a = st.number_input(
            'パラメータα',
            min_value=0.0,
            max_value=100.0,
            value=1.0,
        )
    with col2:   
        b = st.number_input(
            'パラメータβ',
            min_value=0.0,
            max_value=100.0,
            value=2.0,
        )
    x = np.arange(0, 1.02, 0.02)
    y = map(lambda k: k**(a-1)*(1-k)**(b-1)/beta(a,b),x)
    df = pd.DataFrame(data=y,index=x,columns=["y"])
    st.latex(r'''確率密度関数：f(x)=\frac{x^{(α-1)}(1-x)^{(β-1)}}{B(α,β)}''')
    st.latex(r'''平均：\frac{α}{α+β}　　分散：\frac{αβ}{(α+β)^2(α+β+1)}''')
    st.line_chart(df)
