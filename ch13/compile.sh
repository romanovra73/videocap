#!/bin/sh
g++ -o svrd_ch13 chan13.cpp ../include/dvr.cpp ../include/myCV.cpp -std=c++17 -lpqxx -lpq -lcurl `pkg-config --cflags --libs opencv4`
