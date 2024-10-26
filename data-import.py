import logging
import os
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataImporter:
    def __init__(
        self,
        host="127.0.0.1",
        user="root",  # Hardcoded username
        password="Mysql@1031",  # Hardcoded password
        database="customer_orders_db",
        port=3306,
    ):
        """Initialize database connection"""
        try:
            # URL encode the password to handle special characters
            encoded_password = quote_plus(password)

            # Create connection URL for MySQL with IPv6
            connection_url = (
                f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}"
                "?charset=utf8mb4"
            )

            logger.info(f"Attempting to connect to database at {host}:{port}")

            # Test connection before assigning to self
            self.engine = create_engine(
                connection_url,
                connect_args={
                    "connect_timeout": 5,
                    "read_timeout": 30,
                    "write_timeout": 30,
                },
            )

            # Verify connection works
            # with self.engine.connect() as conn:
            #     conn.execute("SELECT 1")
            #     logger.info("Database connection established successfully")

        except OperationalError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database connection: {str(e)}")
            raise

    def import_customers(self, filepath):
        """Import customers data from CSV"""
        try:
            df = pd.read_csv(filepath)
            df.to_sql("customers", self.engine, if_exists="append", index=False)
            logger.info(f"Successfully imported {len(df)} customer records")
        except Exception as e:
            logger.error(f"Error importing customers: {str(e)}")
            raise

    # def import_orders(self, filepath):
    #     """Import orders data from CSV"""
    #     try:
    #         df = pd.read_csv(filepath)
    #         df["created_at"] = pd.to_datetime(df["created_at"])
    #         df.to_sql("orders", self.engine, if_exists="append", index=False)
    #         logger.info(f"Successfully imported {len(df)} order records")
    #     except Exception as e:
    #         logger.error(f"Error importing orders: {str(e)}")
    #         raise

    def import_orders(self, filepath):
        """Import orders data from CSV, excluding orders with invalid customer IDs."""
        try:
            df = pd.read_csv(filepath)
            df["created_at"] = pd.to_datetime(df["created_at"])

            # Retrieve valid customer IDs from the database
            valid_customer_ids = pd.read_sql(
                "SELECT customer_id FROM customers", self.engine
            )
            valid_customer_ids = set(valid_customer_ids["customer_id"])

            # Filter orders to only include rows with valid customer_id values
            df = df[df["customer_id"].isin(valid_customer_ids)]

            # Insert filtered orders into database
            df.to_sql("orders", self.engine, if_exists="append", index=False)
            logger.info(f"Successfully imported {len(df)} valid order records")
        except Exception as e:
            logger.error(f"Error importing orders: {str(e)}")
            raise


def main():
    try:
        # Initialize importer (will use hardcoded credentials)
        importer = DataImporter()

        # # Import customers
        # customers_file = "customers.csv"
        # if os.path.exists(customers_file):
        #     importer.import_customers(customers_file)
        # else:
        #     logger.error(f"Customers file not found: {customers_file}")
        #
        # Import orders
        orders_file = "order.csv"
        if os.path.exists(orders_file):
            importer.import_orders(orders_file)
        else:
            logger.error(f"Orders file not found: {orders_file}")

    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
