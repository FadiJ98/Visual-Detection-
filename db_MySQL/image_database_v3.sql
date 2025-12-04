-- Image Database (Version 3)
-- Created by: Yousif Pata
-- Added feedback and analytics tables for AI performance tracking

CREATE DATABASE IF NOT EXISTS image_database;
USE image_database;

CREATE TABLE model_feedback (
    feedback_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT,
    predicted_emotion ENUM('Angry', 'Sad', 'Joyful', 'Confused', 'Neutral'),
    actual_emotion ENUM('Angry', 'Sad', 'Joyful', 'Confused', 'Neutral'),
    feedback_text TEXT,
    verified_by VARCHAR(100),
    verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE emotion_statistics (
    emotion ENUM('Angry', 'Sad', 'Joyful', 'Confused', 'Neutral') PRIMARY KEY,
    total_predictions INT DEFAULT 0,
    correct_predictions INT DEFAULT 0,
    accuracy DECIMAL(5,2)
);

DELIMITER //
CREATE TRIGGER update_accuracy
AFTER INSERT ON model_feedback
FOR EACH ROW
BEGIN
    UPDATE emotion_statistics
    SET total_predictions = total_predictions + 1,
        correct_predictions = correct_predictions + (NEW.predicted_emotion = NEW.actual_emotion),
        accuracy = (correct_predictions / total_predictions) * 100
    WHERE emotion = NEW.predicted_emotion;
END;
//
DELIMITER ;
