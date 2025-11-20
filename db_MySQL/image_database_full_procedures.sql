
DROP DATABASE IF EXISTS image_database_basic;
CREATE DATABASE image_database_basic;
USE image_database_basic;

-- 1. Persons: basic info about each person
CREATE TABLE persons (
    person_id   INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    gender      ENUM('Male', 'Female', 'Non-binary', 'Other') NULL,
    emotion     ENUM('Angry', 'Sad', 'Joyful', 'Confused', 'Neutral') NULL,
    location    VARCHAR(100) NULL,
    lighting    VARCHAR(50) NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Images: portrait images linked to a person
CREATE TABLE images (
    image_id    INT AUTO_INCREMENT PRIMARY KEY,
    person_id   INT NOT NULL,
    pose        VARCHAR(50) NULL,
    file_path   VARCHAR(255) NOT NULL,
    -- kept tags simple, but you can remove this column if you want it even smaller
    tags        VARCHAR(255) NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_images_person
        FOREIGN KEY (person_id)
        REFERENCES persons(person_id)
        ON DELETE CASCADE
);

-- 3. Group scenes: info about a group photo / scene
CREATE TABLE group_scenes (
    scene_id      INT AUTO_INCREMENT PRIMARY KEY,
    location      VARCHAR(100) NOT NULL,
    lighting      VARCHAR(50) NULL,
    people_count  INT NOT NULL,
    -- simple tags as free text (no triggers, no auto-generation)
    tags          VARCHAR(255) NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Group members: link people to scenes, and store their role
CREATE TABLE group_members (
    member_id     INT AUTO_INCREMENT PRIMARY KEY,
    scene_id      INT NOT NULL,
    person_id     INT NOT NULL,
    role_in_scene VARCHAR(50) NULL,
    added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_members_scene
        FOREIGN KEY (scene_id)
        REFERENCES group_scenes(scene_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_members_person
        FOREIGN KEY (person_id)
        REFERENCES persons(person_id)
        ON DELETE CASCADE
);
