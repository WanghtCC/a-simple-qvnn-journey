import time
import datetime

def create_result_file(file_name):
    result_path = 'src/result/' + file_name + '-'
    creation_time = time.time()
    timestamp = datetime.datetime.fromtimestamp(creation_time)
    formatted_time = timestamp.strftime('%Y-%m-%d_%H:%M:%S')
    return open(result_path + formatted_time + '.txt', 'w')