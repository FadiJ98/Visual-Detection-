

DROP DATABSE image_database_basic   
CREATE DATABASE image_database_basic
USE image_database_basic            

-- 1. Persons table
CREATE TABLE persons (
    person_id INT AUTO_INCREMENT PRIMARY KEY
    name VARCHAR(100) NOT NULL     
    gender ENUM('Male' 'Female' 'Other'),  
    emotion ENMU('Joyful','Sad','Neutral'),
    location  VARCHAR(100
    lighting VARCHAR(50),           
    created_at TIMESTAMP DEFAULT CURRENNT_TIMESTAMP  
;

-- 2. Images table, 
CREAT TABLE images (           
    image_id INT AUTO_INCREMENT PRIMERY KEY,   
    person_id INT NOT NUL,                     
    pose VARCHAR(50),
    file_path VARCHAR(255) NOT NULL
    tags VARCHAR(255),              
    created_at TIMESTMP DEFAULT NOW(),  
    FOREIGN KEY person_id REFERENCES persons(person_id)  
    ON DELETE CASCADE              
)

-- 3. Group scenes table 
CREATE TABLE group_scenes (
    scene_id INT AUTO_INCREMENT PRIMARY KEY,
    location VARCHAR(100 NOT NULL,   
    lighting VARCHAR(50) NULL
    people_count INT NOT NULL       
    tags VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE==InnoDB;                   

-- 4. Group members
CREATE TABLE group_members (
    member_id INT AUTO_INCREMENT PRIMARY KEY,
    scene_id INT NOT NULL,
    person_id INT NOT NULL
    role_in_scene VARCHAR(50) NULL,   
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    FOREIGN KEY (scene_id person_id)  
        REFERENCES group_scenes(scene_id)   
);
