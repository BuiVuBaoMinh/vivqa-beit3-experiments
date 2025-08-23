import json

# Path to your log file and the report file
log_file = "/home/21khac.dd/bm/beit3-lora-3/log.txt"
report_file = "training_report.txt"

# Initialize variables to track total time and time per epoch
total_time = 0
epoch_times = []

# Read the log file line by line
with open(log_file, "r", encoding="utf-8") as file:
    for line in file:
        # Parse the JSON log line
        log_entry = json.loads(line.strip())
        
        # Extract the time and accumulate total time
        epoch_time = log_entry["time"]
        total_time += epoch_time
        
        # Store the time for each epoch
        epoch_times.append((log_entry["epoch"], epoch_time))

# Write the results to the report file
with open(report_file, "w", encoding="utf-8") as report:
    # Write the total time
    report.write(f"Total time: {total_time:.2f} seconds\n\n")
    
    # Write the time per epoch
    report.write("Time per epoch:\n")
    for epoch, epoch_time in epoch_times:
        report.write(f"Epoch {epoch}: {epoch_time:.2f} seconds\n")

print(f"Report has been saved to {report_file}")
