FROM 192.168.0.108:8082/python@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

COPY bin/ /opt/terminal-agent/bin/

RUN chmod 0555 /opt/terminal-agent/bin/*.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
