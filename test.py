import threading
import time

event = threading.Event()
counter_lock = threading.Lock()
counter = 0

def thread_function(thread_id, event, num_cycles):
    global counter
    for cycle in range(num_cycles):
        print(f"Thread {thread_id} is sleeping for {cycle + 1} time(s).")
        time.sleep(3)
        print(f"Thread {thread_id} finished sleep {cycle + 1}. Waiting for sync.")
        with counter_lock:
            counter += 1
            if counter == num_threads:
                event.set()
        event.wait()
        event.clear()
        with counter_lock:
            counter -= 1
        print(f"Thread {thread_id} proceeding to next cycle.")

def main():
    global num_threads
    num_threads = 10
    num_cycles = 5
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=thread_function, args=(i, event, num_cycles))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    print("All threads have finished.")

if __name__ == "__main__":
    main()
