# Python3

import requests
import datetime
import json
import time
import os

# Helper Methods
def next_week(current_week):
    return current_week + datetime.timedelta(days=7)

def prev_week(current_week):
    return current_week - datetime.timedelta(days=7)

def get_timestamps():
    cur = datetime.datetime.now()
    while True:
        yield (int(cur.timestamp()), int(next_week(cur).timestamp()))
        cur = prev_week(cur)

# Download and cache all timeseries from the Citadel API
def get_timeseries_from_api():
    data_dir = 'data/'

    # First get information about all points in Citadel
    r = requests.get('https://citadel.ucsd.edu/api/point/')
    uuids = sorted([point['uuid'] for point in r.json()['point_list']])

    # Get time series data for every Citadel point
    for i, uuid in enumerate(uuids):
        print('\n\n', flush=True)
        print(str(i) + '\t' + uuid, flush=True)
        
        timestamps = get_timestamps()
        url = 'https://citadel.ucsd.edu/api/point/' + uuid + '/timeseries'  
        sensor_data = {'data': {}}

        num_empty = 0
        for start_time, end_time in timestamps:
            time.sleep(1)
            r = requests.get(url, params={'start_time': start_time, 'end_time': end_time})
            sensor_data['data'].update(r.json()['data'])
            print('\t' + str(len(r.json()['data'])), flush=True)
            
            if len(r.json()['data']) == 0:
                num_empty += 1
            elif len(r.json()['data']) == 10000:
                print('Found 10000 limit', flush=True)
                return
            else:
                num_empty = 0

            if num_empty >= 20:
                break

        f = open(data_dir + uuid + '.json', 'w')
        json.dump(sensor_data, f)
        f.close()

if __name__ == '__main__':
    # Create data directory if it doesn't already exist
    if not os.path.exists('data/'):
        os.makedirs('data/')

    get_timeseries_from_api()
