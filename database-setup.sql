-- Create database
CREATE DATABASE IF NOT EXISTS customer_orders_db;
USE customer_orders_db;

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) ,
    email VARCHAR(100)
);

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    id VARCHAR(10) PRIMARY KEY,
    display_order_id VARCHAR(10) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    created_at DATETIME NOT NULL,
    customer_id VARCHAR(10) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
