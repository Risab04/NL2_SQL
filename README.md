NL2SQL Bot
📌 Overview
NL2SQL Bot is an AI-powered chatbot that converts natural language questions into SQL queries. The project involves creating a custom dataset of natural language (NL) questions and corresponding SQL queries across multiple databases and different role-play scenarios. The model is then fine-tuned to understand diverse query structures and produce accurate SQL outputs for varied database schemas.

This bot makes it easier for non-technical users to interact with databases by simply typing their questions in everyday language.

🚀 Features
Custom Dataset Creation – Built a dataset mapping NL questions to SQL queries for different domains.

Multiple Database Support – Queries generated for various database schemas.

Role-Play Scenarios – Examples designed for different user personas (e.g., librarian, e-commerce manager, automotive analyst).

Fine-Tuned Model – Trained an LLM on the custom dataset for better accuracy.

Error Handling – Ensures valid SQL generation and handles invalid queries gracefully.

Easy Integration – Can be connected to any SQL-supported database.

📂 Dataset
The dataset contains:

Natural Language Questions – Written in multiple tones and complexities.

SQL Queries – Covering SELECT, JOIN, GROUP BY, ORDER BY, aggregation, filtering, and more.

Database Variations – Includes multiple schema structures for robustness.

Role Play – Queries framed as if coming from different roles (manager, customer, analyst, etc.).

🛠️ Tech Stack
Language: Python

Libraries: pandas, transformers, datasets

Model: Fine-tuned LLM (Hugging Face Transformers)

Database: MySQL / PostgreSQL (used for testing)

📈 Training Process
Data Preparation – Curated NL-SQL pairs for multiple databases.

Preprocessing – Tokenized and formatted data for training.

Fine-Tuning – Used Hugging Face Trainer API to fine-tune a pre-trained model.

Evaluation – Tested on unseen NL questions to check accuracy.

🔮 Future Enhancements
Support for multiple SQL dialects.

Handle nested queries and stored procedures.

Add multi-turn conversation support.

