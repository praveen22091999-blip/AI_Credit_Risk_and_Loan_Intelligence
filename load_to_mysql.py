"""
Load synthetic credit risk CSVs into MySQL
Run this AFTER you've created the database + tables in MySQL Workbench.
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error

# ---------- CONFIG: Aiven cloud MySQL (one-time migration) ----------
DB_CONFIG = {
    "host": "mysql-ab4286a-praveen22091999-9a9e.a.aivencloud.com",
    "port": 13243,
    "user": "avnadmin",
    "password": "AVNS_KjCT1GeBga-rU9jw-ow",
    "database": "defaultdb"
}

CUSTOMERS_CSV = "customers.csv"
CREDIT_HISTORY_CSV = "credit_history.csv"
LOANS_CSV = "loans.csv"
# -------------------------------------------


def load_csv_to_table(cursor, df, table_name, columns):
    """Insert a dataframe into a MySQL table in chunks."""
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    insert_query = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

    data = [tuple(row[col] for col in columns) for _, row in df.iterrows()]

    chunk_size = 500
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        cursor.executemany(insert_query, chunk)
        print(f"  Inserted rows {i} to {i + len(chunk)} into {table_name}")


def main():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("Connected to MySQL successfully.\n")

        # ---------- Load customers ----------
        print("Loading customers.csv ...")
        customers = pd.read_csv(CUSTOMERS_CSV)
        load_csv_to_table(
            cursor, customers, "customers",
            ["customer_id", "age", "income", "employment_type", "dependents"]
        )
        conn.commit()
        print("customers table loaded.\n")

        # ---------- Load credit_history ----------
        print("Loading credit_history.csv ...")
        credit_history = pd.read_csv(CREDIT_HISTORY_CSV)
        load_csv_to_table(
            cursor, credit_history, "credit_history",
            ["customer_id", "credit_score", "credit_utilization", "debt_ratio",
             "open_credit_lines", "past_due_30_59", "past_due_60_89",
             "past_due_90plus", "real_estate_loans"]
        )
        conn.commit()
        print("credit_history table loaded.\n")

        # ---------- Load loans ----------
        print("Loading loans.csv ...")
        loans = pd.read_csv(LOANS_CSV)
        load_csv_to_table(
            cursor, loans, "loans",
            ["customer_id", "loan_amount", "tenure_months", "interest_rate",
             "loan_type", "application_date", "default_flag"]
        )
        conn.commit()
        print("loans table loaded.\n")

        # ---------- Verify ----------
        for table in ["customers", "credit_history", "loans"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Row count in {table}: {count}")

        cursor.close()
        conn.close()
        print("\nAll data loaded successfully. MySQL connection closed.")

    except Error as e:
        print(f"MySQL Error: {e}")


if __name__ == "__main__":
    main()
