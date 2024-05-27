FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt update

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

COPY streamlit_app.py /app

EXPOSE 8080

ENTRYPOINT ["streamlit", "run"]

CMD ["streamlit_app.py", "--server.port=8080"]
