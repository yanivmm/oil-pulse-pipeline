FROM astrocrpublic.azurecr.io/runtime:3.1-14

# Install Java (required for PySpark)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Copy scripts and models into the image
COPY scripts/ /usr/local/airflow/scripts/
COPY models/ /usr/local/airflow/models/
USER astro
