FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_loader.py gp_tree.py gp_core.py gp_load_forecasting.py ./

# Mount your data folder at runtime with -v
# e.g. docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/output:/app/output" cos710-a2
VOLUME ["/app/data", "/app/output"]

ENTRYPOINT ["python", "gp_load_forecasting.py"]

# Default: previous_days, 10 runs, output to /app/output
CMD [ \
    "--data", "data/Residential_Energy_Dataset_UK- 2014-2020.csv", \
    "--target-col", "Electricity_load", \
    "--start-row", "0", \
    "--max-rows", "20000", \
    "--mode", "previous_days", \
    "--lag-count", "7", \
    "--points-per-day", "0", \
    "--runs", "10", \
    "--P", "120", \
    "--R", "0.85,0.15", \
    "--S", "4", \
    "--Dm", "9", \
    "--DI", "4", \
    "--Du", "4", \
    "--Sg", "30", \
    "--Dg", "9", \
    "--Tg", "0.75", \
    "--Wg", "10", \
    "--DIg", "3", \
    "--generations", "60", \
    "--out-dir", "output" \
]
