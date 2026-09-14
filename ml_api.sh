#!/usr/bin/env bash
source /opt/python/env/bin/activate
cd /opt/python/ml_api
uvicorn main:app ##--log-config /opt/python/ml_api/logger.yaml
