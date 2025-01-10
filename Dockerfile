FROM python:3.10

RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY predictions predictions
COPY models/model_C_1_0.bin model.bin

ENV MODEL_PATH model.bin
ENV VERSION 2025-01-v01

EXPOSE 9696

ENTRYPOINT [ \
    "gunicorn", \ 
    "--bind=0.0.0.0:9696", \ 
    "predictions.serve:app" \ 
]