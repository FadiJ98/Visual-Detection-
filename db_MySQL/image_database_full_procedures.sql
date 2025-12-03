DROP DATABASE IF EXISTS image_database;
CREATE DATABASE image_database;
USE image_database;

-- ==========================================================
-- 1. Persons Table
-- ==========================================================
CREATE TABLE persons (
    person_id   INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    gender      ENUM('Male', 'Female', 'Other') NULL,
    emotion     ENUM('Angry', 'Sad', 'Joyful', 'Confused', 'Neutral') NULL,
    location    VARCHAR(100) NULL,
    lighting    VARCHAR(50) NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 2. Images Table
-- ==========================================================
CREATE TABLE images (
    image_id    INT AUTO_INCREMENT PRIMARY KEY,
    person_id   INT NOT NULL,
    pose        VARCHAR(50),
    file_path   VARCHAR(255) NOT NULL,
    tags        VARCHAR(255),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
        ON DELETE CASCADE
);

-- ==========================================================
-- 3. Group Scenes Table
-- ==========================================================
CREATE TABLE group_scenes (
    scene_id      INT AUTO_INCREMENT PRIMARY KEY,
    location      VARCHAR(100) NOT NULL,
    lighting      VARCHAR(50),
    people_count  INT NOT NULL,
    tags          VARCHAR(255),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 4. Group Members Table (Many-to-Many Relationship)
-- ==========================================================
CREATE TABLE group_members (
    member_id     INT AUTO_INCREMENT PRIMARY KEY,
    scene_id      INT NOT NULL,
    person_id     INT NOT NULL,
    role_in_scene VARCHAR(50),
    added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scene_id) REFERENCES group_scenes(scene_id)
        ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
        ON DELETE CASCADE
);

-- ==========================================================
-- 5. Predicted Emotions Table (from your attached file)
-- ==========================================================
CREATE TABLE predicted_emotions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT,
    emotion ENUM('Angry', 'Sad', 'Joyful', 'Confused', 'Neutral'),
    confidence DECIMAL(5,2),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
        ON DELETE CASCADE
);

-- ==========================================================
-- 6. Trigger for auto-tagging group scenes
-- ==========================================================
DELIMITER //
CREATE TRIGGER auto_tags_group_scenes
BEFORE INSERT ON group_scenes
FOR EACH ROW
BEGIN
  IF NEW.tags IS NULL OR NEW.tags = '' THEN
    SET NEW.tags = CONCAT(
      IF(NEW.location LIKE '%Shop%' OR NEW.location LIKE '%Restaurant%', 'indoor,', 'outdoor,'),
      LOWER(NEW.lighting), ',group-', NEW.people_count, ',', REPLACE(LOWER(NEW.location), ' ', '-')
    );
  END IF;
END;
//
DELIMITER ;

####################################################

-- ----------------------------------------------------------
-- Basic Viewing Queries
-- ----------------------------------------------------------

# View all people
SELECT * FROM persons;

# View all images
SELECT * FROM images;

# View all group scenes
SELECT * FROM group_scenes;

# View all group members
SELECT * FROM group_members;

-- ----------------------------------------------------------
-- People and Their Images
-- ----------------------------------------------------------

SELECT 
    p.person_id,
    p.name,
    i.image_id,
    i.pose,
    i.file_path
FROM persons p
JOIN images i ON p.person_id = i.person_id;

-- ----------------------------------------------------------
-- Scenes and Their Members (with Roles)
-- ----------------------------------------------------------

SELECT 
    g.scene_id,
    g.location,
    g.people_count,
    p.name AS person_name,
    gm.role_in_scene
FROM group_scenes g
JOIN group_members gm ON g.scene_id = gm.scene_id
JOIN persons p ON gm.person_id = p.person_id
ORDER BY g.scene_id;

-- ----------------------------------------------------------
-- Joyful People in Coffee Shops
-- ----------------------------------------------------------

SELECT name, emotion, location
FROM persons
WHERE emotion = 'Joyful' AND location = 'Coffee Shop';

-- ----------------------------------------------------------
-- All Images Tagged “indoor”
-- ----------------------------------------------------------

SELECT * FROM images WHERE tags LIKE '%indoor%';

-- ----------------------------------------------------------
-- Photos Taken Under Warm Indoor Lighting
-- ----------------------------------------------------------

SELECT i.file_path, p.name, i.pose
FROM images i
JOIN persons p ON i.person_id = p.person_id
WHERE i.tags LIKE '%warm%' OR p.lighting = 'Warm Indoor';

-- ----------------------------------------------------------
-- Recently Added People (past 24 hours)
-- ----------------------------------------------------------

SELECT name, created_at
FROM persons
WHERE created_at >= NOW() - INTERVAL 1 DAY;

-- ----------------------------------------------------------
-- Most Recently Added Group Members
-- ----------------------------------------------------------

SELECT p.name, g.location, gm.added_at
FROM group_members gm
JOIN persons p ON gm.person_id = p.person_id
JOIN group_scenes g ON gm.scene_id = g.scene_id
ORDER BY gm.added_at DESC;

-- ----------------------------------------------------------
-- Full Context Query (People + Image + Scene)
-- ----------------------------------------------------------

SELECT
    p.name,
    p.gender,
    p.emotion,
    i.pose,
    i.file_path,
    g.location AS scene_location,
    g.lighting AS scene_lighting,
    gm.role_in_scene
FROM group_members gm
JOIN persons p ON gm.person_id = p.person_id
JOIN group_scenes g ON gm.scene_id = g.scene_id
LEFT JOIN images i ON i.person_id = p.person_id
ORDER BY g.scene_id, p.name;

-- ----------------------------------------------------------
-- Emotion Counts
-- ----------------------------------------------------------

SELECT emotion, COUNT(*) AS total_people
FROM persons
GROUP BY emotion;

-- ----------------------------------------------------------
-- Images Per Person
-- ----------------------------------------------------------

SELECT p.name, COUNT(i.image_id) AS total_images
FROM persons p
LEFT JOIN images i ON p.person_id = i.person_id
GROUP BY p.person_id;

-- ----------------------------------------------------------
-- Scenes With More Than 5 People
-- ----------------------------------------------------------

SELECT scene_id, location, people_count
FROM group_scenes
WHERE people_count > 5;

-- ----------------------------------------------------------
-- Trends (Predicted Emotions)
-- ----------------------------------------------------------

SELECT i.file_path, e.emotion, e.confidence, e.detected_at
FROM predicted_emotions e
JOIN images i ON e.image_id = i.image_id;

-- ----------------------------------------------------------
-- Tag Verification
-- ----------------------------------------------------------

SELECT scene_id, location, lighting, people_count, tags
FROM group_scenes;

SELECT image_id, file_path, tags
FROM images;
