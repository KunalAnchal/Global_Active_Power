from flask import Flask, render_template
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import WriteOptions
import datetime
import threading

# Load your pre-trained model
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

DATASET_PATH = "chillerdata.csv"

# InfluxDB 2.x Configuration
INFLUXDB_TOKEN = "Uxh3_M9yNWhCE-Ne9xeMYV_I0-sGXBBF3KELMTLDT8UUmcT__jUxAPVmzKmF-DC58dJvsBFovQwtYxrT5hOWeg=="
INFLUXDB_ORG = "IIIOT-INFOTECH"
INFLUXDB_BUCKET = "Machine Learning"
INFLUXDB_URL = "http://192.168.1.130:8086/"

# Initialize InfluxDB client with optimized write options
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=WriteOptions(
    batch_size=1000,
    flush_interval=10_000,
    jitter_interval=2_000,
    retry_interval=5_000
))


@app.route('/')
def index():
    return render_template('index.html', graph=None, data_table=None, excel_file=None, error=None)


df = pd.read_csv(DATASET_PATH)


@app.route('/predict', methods=['POST'])
def predict_and_plot():
    try:
        global df

        # Ensure that the CSV file has the expected columns
        required_columns = ['Outside Temperature (F)', 'Dew Point (F)', 'Humidity (%)', 'Wind Speed (mph)',
                            'Pressure (in)', 'Month', 'Day', 'Hour']
        if not set(required_columns).issubset(df.columns):
            return render_template('index.html', error="The CSV file is missing required columns.", graph=None,
                                   data_table=None, excel_file=None)

        # Perform predictions using the model
        df['Prediction'] = model.predict(df[required_columns])

        # Create distribution plots for Load_Type and Prediction
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.figure()

        # Distribution plot for Chilled Water Rate and Prediction with dark mode
        # sns.histplot(df['Chilled Water Rate (L/sec)'], label='Chilled Water Rate (L/sec)', kde=True, color='yellow')
        # sns.histplot(df['Prediction'], label='Prediction', kde=True, color='red')
        plt.title('Distribution of Chilled Water Rate (L/sec) and Prediction', color='white')
        plt.xlabel('Values', color='white')
        plt.ylabel('Density', color='white')
        plt.legend()

        # Customize for dark mode
        # Customizing the plot for clarity
        sns.histplot(df['Chilled Water Rate (L/sec)'], label='Chilled Water Rate (L/sec)', kde=True, color='yellow')
        sns.histplot(df['Prediction'], label='Prediction', kde=True, color='red')

        # Set labels and title
        plt.title('Distribution of Chilled Water Rate (L/sec) and Prediction', color='white')
        plt.xlabel('Values', color='white')
        plt.ylabel('Density', color='white')  # Clarify that this is density or count
        plt.legend()

        # Customize for dark mode
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_facecolor('black')
        plt.xticks(color='white')
        plt.yticks(color='white')  # Optional: Change y-tick labels color for contrast

        # Save the plot
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='black')  # Ensure black background
        img_buffer.seek(0)
        graph = base64.b64encode(img_buffer.read()).decode('utf-8')

        # Generate data tables for the original data and the prediction
        data_table = df.to_html(classes='table table-condensed table-bordered table-striped')

        # Save the predicted data as an Excel file
        excel_file_path = 'predicted_data.xlsx'
        df.to_excel(excel_file_path, index=False)

        # Start a new thread to send data to InfluxDB asynchronously
        threading.Thread(target=send_to_influxdb_continuously, args=(df, 0.01)).start()

        return render_template('index.html', graph=graph, data_table=data_table, excel_file=excel_file_path, error=None)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}", graph=None, data_table=None,
                               excel_file=None)


def send_to_influxdb_continuously(df, delay=0.01):
    """
    Continuously send actual and predicted data to InfluxDB, row by row, with a delay.

    Args:
        df (pd.DataFrame): DataFrame containing the data to send.
        delay (int): Time delay (in seconds) between sending each row.
    """
    try:
        base_time = datetime.datetime.utcnow()  # Base timestamp for unique times

        for index, row in df.iterrows():
            try:
                # Generate a unique timestamp by adding a delay for each row
                timestamp = (base_time + datetime.timedelta(seconds=index * delay)).isoformat()

                # Create a point for each row without row_id tag
                point = Point("chilled_water_data") \
                    .field("actual_chilled_water_rate", float(row['Chilled Water Rate (L/sec)'])) \
                    .field("predicted_chilled_water_rate", float(row['Prediction'])) \
                    .time(timestamp)  # Assign a unique timestamp

                # Write the point to InfluxDB
                write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

                print(f"Row {index} written to InfluxDB at {timestamp}")

                # Introduce a delay to simulate continuous data flow
                time.sleep(delay)

            except Exception as e:
                print(f"Error writing row {index} to InfluxDB: {e}")

        print("All data has been sent to InfluxDB.")

    except Exception as e:
        print(f"Error in send_to_influxdb_continuously: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8045)
