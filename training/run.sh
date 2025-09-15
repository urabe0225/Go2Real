#!/bin/bash
xhost +local:
docker compose build
docker compose up -d
