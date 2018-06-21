---
layout: default
title: homepage
---
Introduction
============

This repository is for some python and R code I wrote to make comparisons between different kids' mountain bikes that are available in North America this year. There are for 2018 a number of dozen bikes now available in the 24" wheel size at a price and quality level suitable for real trail riding. In addition, there are 26" wheel sized bikes coming to market, as well as extra-small sized 27.5" wheel bikes. Choosing a bike out of this assortment has become more difficult and that's what these scripts are intended to help with.

The raw data for the bikes are located in a [google sheet](https://docs.google.com/spreadsheets/d/1FodMz3A9-ehyC2bdrjs72kN5OgBBWe_H9EtYCfVpBu0/edit?pli=1#gid=1960608829). 

The comparisons are currently being done in two forms, geometry and price/weight regressions.

Geometry
--------

Bicycle geometry has 10 or 11 different parameters, not all of which are independent. Traditionally, the values of these parameters are tabulated by manufacturers, and are compared by buyers to help understand sizing and fit. As a 10 parameter space though, this is not actually that easy to do. 

Price/Weight Regression
-----------------------

Bicycle weight has long been a key factor in selecting a performance mountain bike, along with the price. Despite the importance of the price-weight relationship, bicycle manufacturers have traditionally been loathe to reveal weight information, and yet there shouldn't really be any magic to it. Weight ought to be a very straightforward function of the quality of the components and frame that make up the bike, with price being the measure of quality. If that is true, then it should be possible to construct useful multi-variate regressions for price and weight. 

### Details and Results ###

The details of the geometry and regression analyses are described in more detail in [this](https://docs.google.com/document/d/1GCeHPkG0CdZl3O7KylSOYC2eS9sgoozxHD8z3Y-BDEU/edit) document, along with results.