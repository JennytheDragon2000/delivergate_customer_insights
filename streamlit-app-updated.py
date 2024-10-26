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
    password = "Mysql%401031"  # URL-encoded password
    engine = create_engine(
        f"mysql+pymysql://root:{password}@localhost/customer_orders_db"
    )
    return engine


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


def prepare_customer_features(df):
    """Prepare features for the prediction model"""
    # Calculate features per customer
    customer_features = (
        df.groupby("customer_id")
        .agg(
            {
                "total_amount": ["sum", "mean", "std"],
                "id": "count",
                "created_at": ["min", "max"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    customer_features.columns = [
        "customer_id",
        "total_spent",
        "avg_order_value",
        "order_value_std",
        "order_count",
        "first_order",
        "last_order",
    ]

    # Calculate days between first and last order
    customer_features["customer_age_days"] = (
        customer_features["last_order"] - customer_features["first_order"]
    ).dt.days

    # Calculate average days between orders
    customer_features["avg_days_between_orders"] = (
        customer_features["customer_age_days"] / customer_features["order_count"]
    )

    # Define repeat customer (more than 1 order)
    customer_features["is_repeat"] = customer_features["order_count"] > 1

    return customer_features


def train_prediction_model(features_df):
    """Train the prediction model"""
    # Select features for training
    feature_columns = [
        "total_spent",
        "avg_order_value",
        "order_value_std",
        "order_count",
        "customer_age_days",
        "avg_days_between_orders",
    ]

    X = features_df[feature_columns]
    y = features_df["is_repeat"]

    # Handle any NaN values
    X = X.fillna(0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, feature_columns, X_test_scaled, y_test


def predict_customer_repeat_purchase(model, scaler, features, feature_columns):
    """Predict probability of repeat purchase for a specific customer"""
    features_scaled = scaler.transform(features[feature_columns].fillna(0))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob


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

    # Apply date filters
    filtered_df = orders_df[
        (orders_df["created_at"].dt.date >= start_date)
        & (orders_df["created_at"].dt.date <= end_date)
    ]

    # Prepare customer features and train model
    customer_features = prepare_customer_features(filtered_df)

    # Only proceed with ML if we have enough data
    if len(customer_features) >= 10:
        model, scaler, feature_columns, X_test, y_test = train_prediction_model(
            customer_features
        )
        accuracy = model.score(X_test, y_test)

        # Display model metrics
        st.sidebar.subheader("Model Metrics")
        st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")
    else:
        st.warning(
            "Not enough data for prediction model (minimum 10 customers required)"
        )
        return

    # Main dashboard metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", f"${filtered_df['total_amount'].sum():,.2f}")
    with col2:
        st.metric("Unique Customers", len(filtered_df["customer_id"].unique()))
    with col3:
        st.metric("Total Orders", len(filtered_df))

    # Display filtered data with customer selection
    st.subheader("Customer Orders")

    # Add customer selection
    selected_customer = st.selectbox(
        "Select a customer to predict repeat purchase probability:",
        options=customer_features["customer_id"].unique(),
        format_func=lambda x: f"Customer {x}",
    )

    # Show prediction for selected customer
    if selected_customer:
        customer_data = customer_features[
            customer_features["customer_id"] == selected_customer
        ]

        prob = predict_customer_repeat_purchase(
            model, scaler, customer_data, feature_columns
        )

        # Display customer insights
        st.subheader("Customer Insights")
        cols = st.columns(2)

        with cols[0]:
            st.metric(
                "Repeat Purchase Probability",
                f"{prob:.1%}",
                delta="High" if prob > 0.7 else "Low" if prob < 0.3 else "Medium",
            )

            # Customer statistics
            stats = customer_data.iloc[0]
            st.write(f"Total Orders: {stats['order_count']}")
            st.write(f"Total Spent: ${stats['total_spent']:.2f}")
            st.write(f"Average Order Value: ${stats['avg_order_value']:.2f}")

        with cols[1]:
            st.write("Customer History")
            customer_orders = filtered_df[
                filtered_df["customer_id"] == selected_customer
            ].sort_values("created_at", ascending=False)
            st.dataframe(
                customer_orders[["created_at", "display_order_id", "total_amount"]],
                hide_index=True,
            )

    # Display filtered orders
    st.subheader("All Orders")
    st.dataframe(filtered_df)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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


if __name__ == "__main__":
    main()
