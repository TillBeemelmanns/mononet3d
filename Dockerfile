FROM tensorflow/tensorflow:devel-gpu

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir /.local
RUN chown -R 1000:1000 /.local