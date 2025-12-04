-- Image Database (Version 2)
-- Created by: Yousif Pata
-- Added system_logs and api_requests tables for backend tracking and API monitoring

CREATE DATABASE IF NOT EXISTS image_database;
USE image_database;

CREATE TABLE system_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50),
    operation ENUM('INSERT', 'UPDATE', 'DELETE'),
    record_id INT,
    description TEXT,
    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE api_requests (
    request_id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(255),
    method ENUM('GET', 'POST', 'PUT', 'DELETE'),
    status_code INT,
    response_time_ms DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DELIMITER //
CREATE TRIGGER log_new_prediction
AFTER INSERT ON predicted_emotions
FOR EACH ROW
BEGIN
    INSERT INTO system_logs (table_name, operation, record_id, description)
    VALUES ('predicted_emotions', 'INSERT', NEW.prediction_id, 'New emotion prediction stored.');
END;
//
DELIMITER ;
