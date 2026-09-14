#!/bin/sh
g++ -o chan12 chan12.cpp ../include/dvr.cpp ../include/myCV.cpp -std=c++17 -lpqxx -lpq -lcurl `pkg-config --cflags --libs opencv4`
