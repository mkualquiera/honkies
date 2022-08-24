import multiprocessing

def worker(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue):
    while True:
        job = in_queue.get()
        
        job_id = job["id"]
        job_parameters = job["parameters"]

        # = stuff =
