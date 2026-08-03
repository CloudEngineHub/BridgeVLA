'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''
import json
import os

def calculate_success_rates(task_json_path, result_json_path):
    # Read the task list file
    with open(task_json_path, 'r') as f:
        tasks = json.load(f)
    
    # Read the test result file
    results = []
    with open(result_json_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            results.append(result)

    task_success_rates = []
    total_success_count = 0
    total_trials = 0
    
    # Each task is tested 20 times
    tests_per_task = 20
    num_tasks = len(tasks)
    
    for i in range(num_tasks):
        start_index = i * tests_per_task
        end_index = start_index + tests_per_task
        task_results = results[start_index:end_index]
        
        success_count = sum(1 for res in task_results if res["success"] == 1.0)
        success_rate = success_count / tests_per_task
        task_success_rates.append(success_rate)
        
        total_success_count += success_count
        total_trials += tests_per_task
    
    # Calculate the overall success rate
    overall_success_rate = total_success_count / total_trials
    
    return tasks, task_success_rates, overall_success_rate

def save_results(task_json_path, result_json_path, tasks, task_success_rates, overall_success_rate):
    result_dir = os.path.dirname(result_json_path)
    result_file_name = os.path.basename(result_json_path)
    txt_file_name = os.path.splitext(result_file_name)[0] + '.txt'
    txt_file_path = os.path.join(result_dir, txt_file_name)
    
    with open(txt_file_path, 'w') as f:
        for task, rate in zip(tasks, task_success_rates):
            f.write(f"Task: {task}, Success Rate: {rate * 100:.2f}%\n")
        f.write(f"\nOverall Success Rate: {overall_success_rate * 100:.2f}%\n")

# Define the expected number of lines in the result.json file for each setting
EXPECTED_LINES = {
    'train': 620,
    'test_l2': 560,
    'test_l3': 420,
    'test_l4': 240
}

def main(total_result_path):
    # Please modify these paths according to the actual situation
    task_json_paths = {
        'train': './assets/taskvars_train.json',
        'test_l2': './assets/taskvars_test_l2.json',
        'test_l3': './assets/taskvars_test_l3.json',
        'test_l4': './assets/taskvars_test_l4.json'
    }

    seed_dirs = [d for d in os.listdir(total_result_path) if d.startswith('seed')]
    for seed_dir in seed_dirs:
        seed_path = os.path.join(total_result_path, seed_dir)
        seed_overall_rates = []
        for setting in task_json_paths.keys():
            setting_path = os.path.join(seed_path, setting)
            result_json_path = os.path.join(setting_path, 'result.json')
            if os.path.exists(result_json_path):
                # Check the number of lines in the file
                with open(result_json_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) != EXPECTED_LINES[setting]:
                        error_msg = f"Error: {result_json_path} has {len(lines)} lines, but expected {EXPECTED_LINES[setting]} lines."
                        print(error_msg)
                        raise ValueError(error_msg)

                task_json_path = task_json_paths[setting]
                tasks, task_success_rates, overall_success_rate = calculate_success_rates(task_json_path, result_json_path)
                save_results(task_json_path, result_json_path, tasks, task_success_rates, overall_success_rate)
                seed_overall_rates.append(overall_success_rate)

        # Calculate the average success rate of the four settings under the current seed
        if seed_overall_rates:
            average_success_rate = sum(seed_overall_rates) / len(seed_overall_rates)
            avg_txt_path = os.path.join(seed_path, 'average_success_rate.txt')
            with open(avg_txt_path, 'w') as f:
                f.write(f"Average Success Rate across all settings for this seed: {average_success_rate * 100:.2f}%\n")

if __name__ == "__main__":
    total_result_path = 'PATH_TO_RESULT_DIR'
    main(total_result_path)
