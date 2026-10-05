# Ecommerce Linear Regression Portal

This project is a complete FastAPI web application built around the ecommerce customer linear regression model.

## Features

- User login and logout
- Registration page
- Profile management
- Dashboard with model summary and feature impact chart
- Recent prediction tracking on the dashboard
- Single customer prediction page
- Bulk CSV prediction upload
- Admin panel for user overview and management
- Role updates and user deletion from the admin area
- SQLite-backed user storage

## Run locally

```bash
python -m uvicorn app:app --reload
```

Then open:

- http://127.0.0.1:8000/login
- http://127.0.0.1:8000/docs

## Demo login

- Username: admin
- Password: admin123

## Example prediction request

```json
{
  "avg_session_length": 34.5,
  "time_on_app": 12.7,
  "time_on_website": 39.5,
  "length_of_membership": 4.1
}
```
