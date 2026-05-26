FROM gcr.io/deeplearning-platform-release/base-cu124

WORKDIR .

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

COPY requirements.txt .
COPY run_experiment.py .
COPY quick_experiment.py .
COPY utils.py .
COPY setup.py .
COPY recbole/ recbole/
COPY nlp_cache.pt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --verbose

 ENTRYPOINT ["python", "-m", "run_experiment"]
