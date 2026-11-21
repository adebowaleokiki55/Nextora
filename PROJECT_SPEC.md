# MALARIA MISSION CONTROL — Complete Project Description

We are building **Malaria Mission Control**, an AI-powered malaria early-warning, intelligence and response coordination platform designed to help public-health teams anticipate rising malaria risk, identify vulnerable areas, detect facility shortages and coordinate interventions before situations become worse.

The platform consists of five interconnected modules that operate as one unified system, sharing information through a central database and backend.

**Module 1 — Environmental and Historical Surveillance Intelligence**

This module collects and analyses historical malaria reports, rainfall, temperature, vegetation and other relevant environmental information. Historical malaria reports are records showing how many malaria cases were reported in specific geographical locations over different periods, such as monthly or weekly reports from health facilities or public-health surveillance systems. The purpose is to understand previous malaria trends, identify seasonal patterns and monitor environmental conditions that may favour malaria transmission. The module combines these different data sources to provide a picture of malaria conditions across monitored locations.

**Module 2 — Malaria Prediction Engine**

This module uses machine learning to analyse the information collected by Module 1 and estimate how malaria risk may change over the coming weeks. It examines historical malaria trends, environmental conditions and seasonal patterns to identify locations where malaria activity may increase. It generates location-specific risk scores, forecasts and early-warning alerts. Predictions must be evaluated against historical data and compared with simple forecasting baselines to establish their actual accuracy.

**Module 3 — AI Command Centre and Multi-Agent Coordination**

This is the central intelligence and coordination layer of the platform. It receives risk predictions and alerts from Module 2, examines information from the other modules and coordinates appropriate response activities. It contains specialised AI agents: a Commander Agent that oversees the response, a Surveillance Agent that investigates risk signals, a Supply Agent that checks medicine and testing-kit availability, an Outreach Agent that prepares prevention and awareness messages, and a Policy Agent that generates situation reports and recommendations. These agents work together to prepare coordinated response plans, assign tasks and track progress. Consequential actions, such as dispatching supplies or sending public-health communications, require human approval.

**Module 4 — Medical Supply and Facility Intelligence**

This module monitors malaria-related medical supplies at participating healthcare facilities, including rapid diagnostic test kits and antimalarial medicines. It tracks available stock, consumption and minimum stock thresholds to identify facilities at risk of running out of essential supplies. When a shortage is detected, it calculates the quantity needed, identifies facilities that may have surplus stock and recommends possible redistribution or replenishment. It sends shortage alerts and recommendations to the AI Command Centre so that response teams can prioritise facilities in areas experiencing elevated malaria risk.

**Module 5 — AI Laboratory Intelligence**

This module uses computer vision and deep learning to analyse uploaded malaria blood-smear images. A laboratory worker uploads an image, and a trained convolutional neural network examines the image to classify individual red blood cells as potentially infected or uninfected. The system produces classifications and confidence scores and stores the results alongside the facility, geographical location and date of testing. A qualified laboratory professional must review the results before they are treated as verified laboratory observations. Verified observations can contribute to local malaria surveillance and provide additional information to the prediction engine. The initial implementation uses the NIH Malaria Cell Images dataset, which contains labelled images of individual red blood cells. It should not be presented as an autonomous diagnostic system or a validated parasite-counting tool.

## How All Five Modules Connect

All five modules communicate through one shared backend and central database. They are not five independent applications.

For example, imagine the system is monitoring Ikorodu in Lagos. Module 1 receives historical malaria reports and current rainfall and temperature data, then identifies environmental and surveillance signals that may indicate increasing malaria risk. Module 2 analyses these signals and estimates that malaria risk may rise in Ikorodu over the coming weeks. It generates an early-warning alert that is sent to Module 3.

The AI Command Centre receives the alert and coordinates the other modules. Its Supply Agent requests inventory information from Module 4, which discovers that three participating healthcare facilities have insufficient malaria rapid diagnostic test kits. It calculates the shortages and recommends redistribution from facilities with available surplus stock.

At the same time, Module 5 receives blood-smear images from participating laboratories. It analyses the images, produces cell-level classifications and sends the observations for professional review. Once verified, these laboratory observations become additional surveillance information associated with Ikorodu.

The Command Centre combines the available information into one operational picture: environmental conditions, predicted malaria risk, verified laboratory observations, facility supply levels and outstanding response tasks. It can then prepare recommendations, coordinate approved actions and track the response.

The information also feeds back into the shared system. New verified laboratory observations, subsequent malaria reports, environmental measurements and inventory updates can be used to evaluate and improve future forecasts and response planning.

## Overall Objective

The objective is to build a unified malaria intelligence and response platform that answers five important questions:

1. How accurately can we forecast malaria trends in different locations?
2. How early can we identify geographical areas where malaria risk is increasing?
3. How quickly can we identify healthcare facilities experiencing shortages of essential malaria supplies?
4. How much faster can public-health teams coordinate investigations and responses using the platform?
5. Can combining environmental, surveillance, laboratory and supply information help public-health teams make better-informed decisions?

The platform is intended to support public-health professionals, not replace medical professionals, laboratory scientists or official disease-surveillance systems. Its purpose is to transform scattered information into actionable intelligence and coordinated responses.

## Proposed Technology Stack

- Frontend: React, Vite and Tailwind CSS.
- Backend: Python and FastAPI.
- Database: PostgreSQL with PostGIS for geographical data.
- Machine learning: Pandas, NumPy, Scikit-learn and XGBoost or LightGBM.
- Computer vision: PyTorch and OpenCV.
- Maps: Leaflet.
- Visualisations: Plotly.
- AI agents: Claude API or an appropriate agent framework.
- Environmental data: rainfall, temperature and satellite-derived environmental datasets.
- Malaria data: available historical surveillance records, WHO datasets and Malaria Atlas Project datasets.

Build the platform as one integrated application with five clearly separated modules, a shared data model, a common alert system and a unified command-centre dashboard. Use real data wherever accessible, clearly label synthetic demonstration data, preserve the source and verification status of every observation, and never present simulated results as real-world achievements.