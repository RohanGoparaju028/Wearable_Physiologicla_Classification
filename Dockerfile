FROM python:latest 

WORKDIR /app 

COPY requirments.txt .

RUN pip install -r requirments.txt 

COPY . . 

CMD ["python", "main.py"]