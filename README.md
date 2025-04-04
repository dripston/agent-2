# Weather Data Aggregation Agent

A Python-based weather data aggregation system that collects and processes weather information from multiple sources.

## Features

- Multi-source weather data fetching
- Weighted confidence-based data aggregation
- 24-hour weather predictions
- Support for multiple weather parameters:
  - Temperature
  - Humidity
  - Pressure
  - Wind Speed
  - Wind Direction

## Currently Active Sources

- OpenMeteo (Primary source)
- NOAA Weather Service
- GOES Satellite Data

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install requests