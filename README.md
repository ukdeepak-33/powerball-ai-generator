# powerball-ai-generator
An AI-powered Powerball number generator that analyzes historical trends
# Powerball AI Generator

An intelligent API that generates Powerball numbers based on historical data analysis and machine learning.

## Features

- Smart number generation based on historical trends
- Group A number analysis
- Odd/even ratio tracking
- Consecutive number detection
- RESTful API built with FastAPI

## API Endpoints

- `GET /` - Health check and status
- `GET /generate` - Generate new Powerball numbers with analysis
- `GET /analyze` - Get historical trend analysis
- `GET /health` - Service health check

## Deployment

This application is configured for deployment on Render.com. Environment variables needed:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase anon/key

## Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env` file
3. Run: `uvicorn api.index:app --reload`
