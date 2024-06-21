import streamlit as st
import os
from groq import Groq
import random
import sys
import io
import pandas as pd
from custom_agents.private_docs_agent import private_docs_agent
from langchain_groq import ChatGroq


api_key = os.environ['GROQ_API_KEY']
chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")
print("#33 ",api_key,chat)



@st.cache_resource
def get_private_docs_agent(_llm):
    return private_docs_agent(_llm)

private_docs_helper = get_private_docs_agent(chat)

def f_preguntar():
    pass #st.title("####1")


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    global info_manager
    # Get Groq API key
    


    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    #with col:  
    #    st.image('groqcloud_darkmode.png')


    
    # The title and greeting message of the Streamlit application
    st.title("Private Multidoc Chat - Ask and expert about your data")
    st.write("Chat with private docs (about stock market)")
    
    # Add customization options to the sidebar
    st.sidebar.title('Available experts:')
    st.sidebar.write('Private document expert with multiple documents (about stock market)')
    st.sidebar.write('Example 1: best usage of candlestick charts?')
    st.sidebar.write('Example 2: how to calculate liquidity risk?')
    st.sidebar.write('Example 3: strategies about intraday trading?')
    st.sidebar.write('Example 4: approaches to time series in investments?')
    st.sidebar.write('Example 5: notes on portfolio diversification')
    st.sidebar.write('Example 6: information about pricing an IPO')
    st.sidebar.write('Example 7: notes about picking right stocks')
    st.sidebar.write('Document source: https://the-eye.eu/public/Books/cdn.preterhuman.net/texts/finance_and_marketing/stock_market/')
    message = st.text_input("Ask the expert?:",on_change=f_preguntar,key = "userq")

    if message:
        
        st.session_state.chat_history=[]
        st.session_state.chat_history.append(message)
        #query = "check the internal microsoft report about revenue figures, add more recent values from internet and the plot it"
        query = message
        generation_log = f"Step: Init agent\n"
        info = "Step: Checking if question is about finance or investments\n"
        generation_log += info

        output = private_docs_helper.ask_question({"question": query,"generation_log":generation_log})
        #print("=======FINAL OUTPUT========", output["generation"])
        message_data = output
        

        st.write(message_data)

       
        st.session_state.chat_history.append(message_data)
  
    st.write ("<-- Links to the documents on the sidebar.")
    st.write ("List of indexed documents:")
    st.write ("""
10 Minute Guide To Investing In Stocks.pdf\n
17 Money Making Candle Formations.pdf\n
17 Money Making Candlestick Formations.pdf\n
25 Rules Of Trading.pdf\n
A Comparison Of Dividend Cash Flow And Earnings Approaches To Equity Valuation.pdf\n
Admati And Pfleiderer-A Theory Of Intraday Patterns - Volume And Price Variability.pdf\n
Aggarwal And Conroy-Price Discovery In Initial Public Offerings And The Role Of The Lead Underwriter.pdf\n
Alan Farley - 3 Swing Trading Examples, With Charts, Instructions, And Definitions To Get You Sta.pdf\n
Alan Farley - Pattern Cycles - Mastering Short-Term Trading With Technical Analysis (Traders' Library).pdf\n
Anshumana And Kalay-Can Splits Create Market Liquidity - Theory And Evidence.pdf\n
Application Of Multi-Agent Games To The Prediction Of Financial Time-Series.pdf\n
Aust Vs Int'l Equity Portfolio Journal.pdf\n
Barbara Star - Hidden Divergence.pdf\n
Basic Financial Strategies.pdf\n
Bessembinder And Venkataraman-Does An Electronic Stock Exchange Need An Upstairs Market.pdf\n
Big Profit Patterns Using Candlestick Signals And Gaps - Stephen W  Bigalow.pdf\n
Borsellino Lewis 2001 - Trading Es And Nq Futures Course.pdf\n
Breman And Subrahmanyam-Investment Analysis And Price Formation In Securities Markets.pdf\n
Building Your E-Mini Trading Strategy - Giuciao Atspace Org.pdf\n
Chan, Chockalingam And Lai-Overnight Information And Intraday Trading Behavior - Evidence From Nyse Cross.pdf\n
Choosing A Trading System That Actually Works.pdf\n
Chordia, Roll And Subrahmanyam -Commonality In Liquidity.pdf\n
Chordia, Sarkar And Subrahmanyam -An Empirical Analysis Of Stock And Bond Market Liquidity.pdf\n
Client Expectations.pdf\n
Combining Bollinger Bands & Rsi.pdf\n
Common Sense Commodities A Common Sense Approach To Trading Commodities.pdf\n
Competition Between Exchanges Euronext Versus Xetra.pdf\n
Cynthia Kase - Multi-Dimensional Trading.pdf\n
Daniel A Strachman - Essential Stock Picking Strategies.pdf\n
Day Trading Basket Stocks - Underground Trader.pdf\n
De Matos And Fernandes-Testing The Markov Property With Ultra-High Frequency Financial Data.pdf\n
Demarchi And Foucault-Equity Trading Systems In Europe - A Survey Of Recent Changes.pdf\n
Dennis D Peterson - Developing A Trading System Combining Rsi & Bollinger Bands.pdf\n
Deutsche Bank - Asset Valuation Allocation Models 2001.pdf\n
Deutsche Bank - Asset Valuation Allocation Models 2002.pdf\n
Dow.pdf\n
Economics - How The Stock Market Works.pdf\n
Eday - Trading In Mind.pdf\n
Emotion Free Trading Book.pdf\n
Engle And Lange-Predicting Vnet - A Model Of The Dynamics Of Market Depth.pdf\n
Exchange Rules For The Frankfurt Stock Exchange.pdf\n
F  E  James Jr - Monthly Moving Averages  An Effective Investment Tool .pdf\n
Fernando-Commonality In Liquidity-Transmission Of Liquidity Shocks Across Investors And Securities.pdf\n
Foucault And Kadan-Limit Order Book As A Market For Liquidity.pdf\n
Foucault And Lescourret-Information Sharing, Liquidity And Transaction Costs In Floor-Based Trading Systems.pdf\n
Foucault, Kadan And Kandel-Limit Order Book As A Market For Liquidity.pdf\n
Frino, Mcinish And Toner-The Liquidity Of Automated Exchanges - New Evidence From German Bund Futures.pdf\n
Futures Magazine - The Art Of Day-Trading.pdf\n
Gann - How To Trade.pdf\n
Giot And Grammig-How Large Is Liquidity Risk In An Automated Auction Market.pdf\n
Griffiths, Turnbullb And White-Re-Examining The Small-Cap Myth Problems In Portfolio Formation And Liquidation.pdf\n
Guide To Effective Daytrading-Wizetrade.pdf\n
Hamao And Hasbrouck-Securities Trading In The Absence Of Dealers - Trades, And Quotes On The Tokyo Stock Exchange.pdf\n
Harris, Sofianos And Shapiro-Program Trading And Intraday Volatility.pdf\n
Hartmann, Manna And Manzanares-The Microstructure Of The Euro Money Market.pdf\n
Hollifield, Miller, Sandas And Slive-Liquidity Supply And Demand In Limit Order Markets.pdf\n
How Large Is Liquidity Risk In An Automated.pdf\n
How The Stock Market Works.pdf\n
How To Make Money Shorting Stocks In Up And Down Markets.pdf\n
How to read charts.pdf\n
""")
        
    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            print("#78", message)
            #st.write(f"{message['role']}: {message['content']}" )



if __name__ == "__main__":
    main()





