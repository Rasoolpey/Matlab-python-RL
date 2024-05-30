import threading
import time

# Define the number of threads
num_threads = 10
num_cycles = 5

# Function to be run by each thread
def thread_function(thread_id, barrier, num_cycles):
    for cycle in range(num_cycles):
        # Simulate work with sleep
        print(f"Thread {thread_id} is sleeping for {cycle + 1} time(s).")
        time.sleep(3)
        
        # After work, wait for all threads to reach the barrier
        print(f"Thread {thread_id} finished sleep {cycle + 1} Waiting for sync.")
        barrier.wait()

# Function to print message when all threads reach the barrier
def barrier_action():
    print("All threads successfully finished one iteration")

# Create the barrier with an action
barrier = threading.Barrier(num_threads, action=barrier_action)

# Main function
def main():
    # Create and start threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=thread_function, args=(i, barrier, num_cycles))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads have finished.")

# Run the main function
if __name__ == "__main__":
    main()
