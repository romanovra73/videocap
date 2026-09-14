#!/bin/sh
g++ -o chan1 chan1.cpp ../include/dvr.cpp ../include/myCV.cpp -std=c++17 -lpqxx -lpq -lcurl `pkg-config --cflags --libs opencv4`
