# DeliverGate Customer Insights Dashboard

This project implements a data engineering solution that imports customer and order data from CSV files into a MySQL database and provides a Streamlit web application for data visualization and analysis.

## Prerequisites

- Python 3.8+
- MySQL Server
- pip (Python package manager)

## Setup Instructions

1. Clone this repository:

```bash
git clone <repository-url>
cd customer-orders-dashboard
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up MySQL database:

- Create a MySQL database using the provided SQL scripts in `database_setup.sql`
- Update the database connection parameters in both `data_import.py` and `app.py`

5. Import data:

```bash
python data_import.py
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

- `database_setup.sql`: SQL script for creating the database schema
- `data_import.py`: Script for importing CSV data into MySQL
- `app.py`: Main Streamlit application
- `requirements.txt`: List of Python dependencies

## Features

- Date range filtering for orders
- Customer filtering based on total spend
- Order count filtering
- Interactive visualizations:
  - Top 10 customers by revenue
  - Revenue over time
  - Key metrics dashboard
- Machine learning model for predicting repeat purchasers

## Dependencies

Create a requirements.txt file with the following contents:

```
streamlit==1.32.0
pandas==2.2.0
sqlalchemy==2.0.27
pymysql==1.1.0
plotly==5.18.0
scikit-learn==1.4.0
python-dotenv==1.0.0
```

## Contact

For any questions or issues, please contact:

- Email: [Your Email]
