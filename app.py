from sklearn import preprocessing
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

final_model = r"C:\Users\maseglo.com\Documents\The Data Science Studio\DS Projects\MarketSegmentation-main\The Market Segmentation Project\final_model.sav"
filename = final_model
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")

st.markdown('<style>body{background-color: #1f77b4;color:white}</style>', unsafe_allow_html=True)
st.title("Global AI Ocean Market Segmentation")

with st.expander("Enter customer data for segmentation:"):
    with st.form("my_form"):
        balance = st.number_input(label='💳 Account Balance', step=0.001, format="%.6f")
        balance_frequency = st.number_input(label='🔄 Balance Frequency', step=0.001, format="%.6f")
        purchases = st.number_input(label='🛒 Total Purchases', step=0.01, format="%.2f")
        oneoff_purchases = st.number_input(label='🛍️ One-off Purchases', step=0.01, format="%.2f")
        installments_purchases = st.number_input(label='📅 Installments Purchases', step=0.01, format="%.2f")
        cash_advance = st.number_input(label='💵 Cash Advance', step=0.01, format="%.6f")
        purchases_frequency = st.number_input(label='📊 Purchases Frequency', step=0.01, format="%.6f")
        oneoff_purchases_frequency = st.number_input(label='🛍️ One-off Purchases Frequency', step=0.1, format="%.6f")
        purchases_installment_frequency = st.number_input(label='📅 Installments Frequency', step=0.1, format="%.6f")
        cash_advance_frequency = st.number_input(label='💵 Cash Advance Frequency', step=0.1, format="%.6f")
        cash_advance_trx = st.number_input(label='🔄 Cash Advance Transactions', step=1)
        purchases_trx = st.number_input(label='🛒 Purchases Transactions', step=1)
        credit_limit = st.number_input(label='💳 Credit Limit', step=0.1, format="%.1f")
        payments = st.number_input(label='💸 Total Payments', step=0.01, format="%.6f")
        minimum_payments = st.number_input(label='💸 Minimum Payments', step=0.01, format="%.6f")
        prc_full_payment = st.number_input(label='📈 Percent Full Payment', step=0.01, format="%.6f")
        tenure = st.number_input(label='📅 Tenure (Months)', step=1)

        data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases,
                 cash_advance, purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency,
                 cash_advance_frequency, cash_advance_trx, purchases_trx, credit_limit, payments, 
                 minimum_payments, prc_full_payment, tenure]]

        submitted = st.form_submit_button("Submit")

if submitted:
    clust = loaded_model.predict(data)[0]
    st.success(f'📊 Data Belongs to Cluster {clust}')

    # Filter data by cluster and visualize
    cluster_df1 = df[df['Cluster'] == clust]

    st.write(f"Showing data for customers in **Cluster {clust}**:")
    st.dataframe(cluster_df1)

    # Plot using plotly for better interactivity
    fig = px.histogram(cluster_df1, x='Cluster', title=f'Cluster {clust} Distribution')
    st.plotly_chart(fig)

    # Seaborn plot (additional option)
    plt.rcParams["figure.figsize"] = (10, 4)
    st.write("Distribution of Features within the Cluster:")
    for c in cluster_df1.drop(['Cluster'], axis=1):
        fig, ax = plt.subplots()
        sns.histplot(cluster_df1[c], kde=True, ax=ax)
        st.pyplot(fig)
