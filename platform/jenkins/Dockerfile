# Use the official Jenkins LTS image with JDK 17
FROM jenkins/jenkins:lts-jdk17

# Switch to root user to install packages
USER root

# Install Docker and Docker Compose
# Note: Mounting docker.sock in Docker Compose is an anti-pattern for production.
# For Kubernetes, the Jenkins Kubernetes plugin is recommended to spin up build agents as pods.
# However, to replicate the Docker Compose setup's capability of running Docker commands,
# we install the Docker CLI here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    docker-compose \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure the jenkins user can run docker commands (optional, but good practice if not using root for all operations)
# In this specific Dockerfile, we'll keep the user as 'jenkins' as per the base image,
# and the docker-compose.yaml mounts docker.sock and docker binaries.
# If running this Dockerfile standalone, you might need to add the jenkins user to the docker group.

# Switch back to the jenkins user
USER jenkins

# The volume for jenkins_home is handled by docker-compose.yaml, so no need to create it here.
# The base image already sets up the jenkins user and home directory.
