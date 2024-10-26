import logging
import os
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataImporter:
    def __init__(
        self,
        host="127.0.0.1",
        user="root",
        password="Mysql@1031",
        database="customer_orders_db",
        port=3306,
    ):
        """Initialize database connection"""
        try:
            encoded_password = quote_plus(password)
            connection_url = (
                f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}"
                "?charset=utf8mb4"
            )

            logger.info(f"Attempting to connect to database at {host}:{port}")

            self.engine = create_engine(
                connection_url,
                connect_args={
                    "connect_timeout": 5,
                    "read_timeout": 30,
                    "write_timeout": 30,
                },
            )

        except OperationalError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database connection: {str(e)}")
            raise

    def clean_dataframe(self, df):
        """Clean DataFrame by replacing NaN values with None and ensuring proper types"""
        # Replace NaN values
        df = df.replace({np.nan: None, "nan": None})

        # Convert customer_id to string and remove decimal points
        if "customer_id" in df.columns:
            df["customer_id"] = (
                df["customer_id"].astype(str).replace(r"\.0$", "", regex=True)
            )

        return df

    def import_customers(self, filepath, update_existing=True):
        """Import customers data from CSV with duplicate handling"""
        try:
            # Read customer data
            df = pd.read_csv(filepath)

            # Clean the dataframe
            df = self.clean_dataframe(df)

            # Get existing customer IDs
            existing_customers = pd.read_sql(
                "SELECT customer_id FROM customers", self.engine
            )
            existing_ids = set(existing_customers["customer_id"].astype(str))

            # Split into new and existing customers
            new_customers = df[~df["customer_id"].isin(existing_ids)]
            existing_customers = df[df["customer_id"].isin(existing_ids)]

            # Insert new customers
            if len(new_customers) > 0:
                new_customers.to_sql(
                    "customers", self.engine, if_exists="append", index=False
                )
                logger.info(
                    f"Successfully imported {len(new_customers)} new customer records"
                )

            # Update existing customers if requested
            if update_existing and len(existing_customers) > 0:
                with self.engine.connect() as conn:
                    for _, row in existing_customers.iterrows():
                        update_dict = {
                            k: v for k, v in row.to_dict().items() if v is not None
                        }
                        if len(update_dict) > 1:
                            set_clause = ", ".join(
                                f"{k} = :{k}"
                                for k in update_dict.keys()
                                if k != "customer_id"
                            )
                            if set_clause:
                                update_query = text(
                                    f"""
                                    UPDATE customers 
                                    SET {set_clause}
                                    WHERE customer_id = :customer_id
                                """
                                )
                                conn.execute(update_query, update_dict)
                    conn.commit()
                logger.info(
                    f"Updated {len(existing_customers)} existing customer records"
                )
            elif len(existing_customers) > 0:
                logger.info(
                    f"Skipped {len(existing_customers)} existing customer records"
                )

            return {
                "total_processed": len(df),
                "new_customers": len(new_customers),
                "existing_customers": len(existing_customers),
            }

        except Exception as e:
            logger.error(f"Error importing customers: {str(e)}")
            raise

    def import_orders(self, filepath):
        """Import orders data from CSV, handling invalid customer IDs"""
        try:
            # Read orders data
            orders_df = pd.read_csv(filepath)
            total_orders = len(orders_df)

            # Clean the dataframe and ensure proper types
            orders_df = self.clean_dataframe(orders_df)

            # Convert created_at to datetime
            orders_df["created_at"] = pd.to_datetime(orders_df["created_at"])

            # Ensure total_amount is decimal
            orders_df["total_amount"] = pd.to_numeric(
                orders_df["total_amount"], errors="coerce"
            )

            # Get valid customer IDs from database
            valid_customer_ids = pd.read_sql(
                "SELECT customer_id FROM customers", self.engine
            )
            valid_customer_ids = set(valid_customer_ids["customer_id"].astype(str))

            # Identify invalid orders
            invalid_orders = orders_df[
                ~orders_df["customer_id"].isin(valid_customer_ids)
            ]
            invalid_count = len(invalid_orders)

            # Filter to only valid orders
            valid_orders = orders_df[orders_df["customer_id"].isin(valid_customer_ids)]
            valid_count = len(valid_orders)

            # Import valid orders
            if valid_count > 0:
                valid_orders.to_sql(
                    "orders", self.engine, if_exists="append", index=False
                )
                logger.info(f"Successfully imported {valid_count} valid order records")

            # Log details about invalid orders
            if invalid_count > 0:
                logger.warning(
                    f"Skipped {invalid_count} orders with invalid customer IDs: "
                    f"{invalid_orders['customer_id'].unique().tolist()}"
                )

                # Save invalid orders to a separate file for review
                invalid_orders.to_csv(
                    f"invalid_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    index=False,
                )

            return {
                "total_orders": total_orders,
                "valid_orders": valid_count,
                "invalid_orders": invalid_count,
            }

        except Exception as e:
            logger.error(f"Error importing orders: {str(e)}")
            raise


def main():
    try:
        importer = DataImporter()

        # Import customers first
        customers_file = "customers.csv"
        if os.path.exists(customers_file):
            results = importer.import_customers(customers_file, update_existing=True)
            logger.info(
                f"Customer import summary:\n"
                f"Total customers processed: {results['total_processed']}\n"
                f"New customers imported: {results['new_customers']}\n"
                f"Existing customers updated: {results['existing_customers']}"
            )
        else:
            logger.error(f"Customers file not found: {customers_file}")
            return

        # Then import orders
        orders_file = "order.csv"
        if os.path.exists(orders_file):
            results = importer.import_orders(orders_file)
            logger.info(
                f"Order import summary:\n"
                f"Total orders processed: {results['total_orders']}\n"
                f"Valid orders imported: {results['valid_orders']}\n"
                f"Invalid orders skipped: {results['invalid_orders']}"
            )
        else:
            logger.error(f"Orders file not found: {orders_file}")

    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
