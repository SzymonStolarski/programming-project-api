# Programming project - REST API for object detection
This project was done for one the CS courses @ the University of Economics in Katowice.

## General overview
This repository contains code that creates an REST API written in `FastAPI` to perform object detection on static images via already trained models available in `TensorFlow Hub`. The prediction requests and results are saved on a `PostgreSQL` database instance deployed on `Heroku`.

## Showcase
The containerized API is running on the `Google Cloud Platform` via the `Cloud Run` service: [https://object-detection-api-bvgfjh3qaq-uc.a.run.app](https://object-detection-api-bvgfjh3qaq-uc.a.run.app). Frontend of this application is deployed on [``Streamlit``](https://share.streamlit.io/szymonstolarski/streamlit-object-detection-app/main/app.py). Repo for the frontend can be found on [``https://github.com/SzymonStolarski/streamlit-object-detection-app``](https://github.com/SzymonStolarski/streamlit-object-detection-app).

## Swagger documentation
`FastAPI` swagger documentation is available on the [`/docs`](https://object-detection-api-bvgfjh3qaq-uc.a.run.app/docs) endpoint.

## Run the development environment
You can run the development environment with `docker-compose`. To make sure everything will work you need to have `Docker` installed on your local machine. While being in the folder with `docker-compose.yaml` in your CLI, just type:
```
docker-compose up --build
```
This will orchestrate an API instance running on `0.0.0.0:8080`, combined with a `PostgreSQL` database.

## Way forward
Possible enhancements in the future:

- set up proper CI/CD pipeline via `GitHub actions`,
- add object detection functionality for videos,
- queue mechanism.