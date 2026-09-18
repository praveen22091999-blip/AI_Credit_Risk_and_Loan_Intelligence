-- Run this on your Aiven MySQL service (defaultdb already exists, no need to create it)
-- Connect using: mysql -h <host> -P <port> -u avnadmin -p --ssl-mode=REQUIRED defaultdb

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    age INT,
    income DECIMAL(12,2),
    employment_type VARCHAR(50),
    dependents INT
);

CREATE TABLE IF NOT EXISTS credit_history (
    history_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(20),
    credit_score INT,
    credit_utilization DECIMAL(6,4),
    debt_ratio DECIMAL(6,4),
    open_credit_lines INT,
    past_due_30_59 INT,
    past_due_60_89 INT,
    past_due_90plus INT,
    real_estate_loans INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS loans (
    loan_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(20),
    loan_amount DECIMAL(12,2),
    tenure_months INT,
    interest_rate DECIMAL(5,2),
    loan_type VARCHAR(50),
    application_date DATE,
    default_flag TINYINT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(20),
    probability_default DECIMAL(6,4),
    risk_category VARCHAR(20),
    expected_loss DECIMAL(12,2),
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
