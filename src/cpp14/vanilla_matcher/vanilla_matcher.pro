TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    conceptmatcher.cpp \
    test_matcher.cpp

HEADERS += \
    catch.hpp \
    conceptmatcher.h
