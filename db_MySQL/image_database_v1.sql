-- Image Database (Version 1)
-- Created by: Yousif Pata
-- Base structure for persons, images, group scenes, and emotion predictions

CREATE DATABASE IF NOT EXISTS image_database;
USE image_database;

CREATE TABLE persons (
    person_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT,
    gender ENUM('Male', 'Female', 'Other'),
    lighting VARCHAR(100),
    location VARCHAR(100),
    tags VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE images (
    image_id INT AUTO_INCREMENT PRIMARY KEY,
    person_id INT,
    file_path VARCHAR(255),
    pose ENUM('Front', 'Left', 'Right', 'Up', 'Down'),
    hash_id CHAR(64),
    tags VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

CREATE TABLE group_scenes (
    scene_id INT AUTO_INCREMENT PRIMARY KEY,
    location VARCHAR(100),
    lighting VARCHAR(100),
    people_count INT,
    hash_id CHAR(64),
    tags VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE group_members (
    scene_id INT,
    person_id INT,
    role_in_scene VARCHAR(50),
    PRIMARY KEY (scene_id, person_id),
    FOREIGN KEY (scene_id) REFERENCES group_scenes(scene_id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

CREATE TABLE predicted_emotions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT,
    emotion ENUM('Angry', 'Sad', 'Joyful', 'Confused', 'Neutral'),
    confidence DECIMAL(5,2),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
