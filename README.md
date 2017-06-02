#AGROVOC autocoder

This project uses Python 3.5.x in order to handle all UTF-8 encoding
issues out of the box.

##Install

Please first read the `data` section for OS-based prerequisites.  Once the 
prerequisites have been fulfilled, please run

    # pip install -r requirements.txt

##Python folders

###data

This directory contains the python classes for scraping the FAO AGROVOC coded text 
documents.  Using the scraping scripts requires either Ubuntu, or MacOS to compile 
the necessary libraries.  Before pip installing from the 
[requirements.txt](src/requirements.txt) please do the following steps depending 
on your OS.

On Ubuntu (note: the # indicates root level privileges):

    > # apt-get install qt5-default libqt5webkit5-dev build-essential python-lxml python-pip xvfb
    
On MacOS:

    > brew install qt
    > easy_install pip
    

###model

Machine learning code goes here.

##MySQL Setup

To connect to a MySQL instance within python:

    1. from utils.database import MySQLDataBase
    2. Pass in either: 1) a connection dictionary from config.py or 2) a custom
       connection in dictionary format, with the same keys as the examples in the
       config.
    
In MySQL create a new database called 'agrovoc_autocode.'  Within that database
create a table for containing the training documents

    CREATE TABLE `agrovoc_autocode`.`agris_data` (
      `id` INT NOT NULL AUTO_INCREMENT,
      `doc_id` VARCHAR(400) NULL,
      `text` NVARCHAR(4000) NULL,
      `codes` VARCHAR(4000) NULL,
      `page` INT NULL,
      `search_term` VARCHAR(100) NULL,
      PRIMARY KEY (`id`));
      
A code lookup table called `agrovoc_terms` in the `agrovoc_autocode` database should be
created from the codes.csv table contained in `\db`