import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine


# Database connection
@st.cache_resource
def init_connection():
    # password = os.environ.get("MYSQL_PASSWORD")
    # password = "Mysql@1031"
    password = "Mysql%401031"  # URL-encoded password
    print(f"passowrd is {password}")
    engine = create_engine(
        f"mysql+pymysql://root:{password}@localhost/customer_orders_db"
    )
    return engine
    # return create_engine("mysql+pymysql://root:your_password@localhost/customer_orders_db")


# Data loading functions
@st.cache_data
def load_data():
    engine = init_connection()
    orders_query = """
    SELECT o.*, c.name as customer_name 
    FROM orders o 
    JOIN customers c ON o.customer_id = c.customer_id
    """
    orders = pd.read_sql(orders_query, engine)
    orders["created_at"] = pd.to_datetime(orders["created_at"])
    return orders


def main():
    st.title("Customer Orders Dashboard")

    # Load data
    orders_df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    min_date = orders_df["created_at"].min()
    max_date = orders_df["created_at"].max()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)

    # Total amount filter
    min_amount = float(orders_df["total_amount"].min())
    max_amount = float(orders_df["total_amount"].max())
    amount_threshold = st.sidebar.slider(
        "Minimum Total Amount",
        min_value=min_amount,
        max_value=max_amount,
        value=min_amount,
    )

    # Order count filter
    customer_order_counts = orders_df.groupby("customer_id").size()
    max_orders = int(customer_order_counts.max())
    min_orders = st.sidebar.number_input(
        "Minimum Number of Orders", min_value=1, max_value=max_orders, value=1
    )

    # Apply filters
    filtered_df = orders_df[
        (orders_df["created_at"].dt.date >= start_date)
        & (orders_df["created_at"].dt.date <= end_date)
    ]

    # Customer metrics
    customer_metrics = (
        filtered_df.groupby("customer_id")
        .agg({"total_amount": "sum", "id": "count"})
        .reset_index()
    )

    # Apply amount and order count filters
    valid_customers = customer_metrics[
        (customer_metrics["total_amount"] >= amount_threshold)
        & (customer_metrics["id"] >= min_orders)
    ]["customer_id"]

    filtered_df = filtered_df[filtered_df["customer_id"].isin(valid_customers)]

    # Main dashboard
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Revenue", f"${filtered_df['total_amount'].sum():,.2f}")
    with col2:
        st.metric("Unique Customers", len(filtered_df["customer_id"].unique()))
    with col3:
        st.metric("Total Orders", len(filtered_df))

    # Display filtered data
    st.subheader("Filtered Orders")
    st.dataframe(filtered_df)

    # Top 10 customers by revenue
    top_customers = (
        filtered_df.groupby("customer_name")["total_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig1 = px.bar(
        top_customers,
        title="Top 10 Customers by Revenue",
        labels={"customer_name": "Customer", "value": "Total Revenue"},
    )
    st.plotly_chart(fig1)

    # Revenue over time
    daily_revenue = (
        filtered_df.groupby(filtered_df["created_at"].dt.date)["total_amount"]
        .sum()
        .reset_index()
    )

    fig2 = px.line(
        daily_revenue,
        x="created_at",
        y="total_amount",
        title="Daily Revenue",
        labels={"created_at": "Date", "total_amount": "Revenue"},
    )
    st.plotly_chart(fig2)

    # Machine Learning Model (Bonus)
    st.subheader("Repeat Purchase Prediction Model")

    if len(filtered_df) > 10:  # Only show if we have enough data
        # Prepare features
        customer_features = customer_metrics.copy()
        customer_features["is_repeat"] = customer_features["id"] > 1

        X = customer_features[["total_amount", "id"]]
        y = customer_features["is_repeat"]

        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Display accuracy
        accuracy = model.score(X_test_scaled, y_test)
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    else:
        st.warning("Not enough data for prediction model")


if __name__ == "__main__":
    main()
