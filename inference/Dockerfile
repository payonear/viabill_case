FROM python:3.8-slim

RUN pip install -U pip
RUN pip install -U pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["app.py", "model.pkl", "scaler.pkl", "ohe.pkl", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]