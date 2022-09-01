import torch.multiprocessing as multiprocessing
from quart import Quart, request, jsonify, send_file
from worker import worker
import torch
import json

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
import os
import asyncio
import random

app = Quart(__name__)

GLOBAL_QUEUE = []

OUT_QUEUES = None
IN_QUEUES = None

STARTED_LOOP = False


@app.route("/api/v1/status")
async def status():
    return jsonify(
        {
            "status": "ok",
            "workers": len(IN_QUEUES),
        }
    )


@app.route("/api/v1/enqueue")
async def enqueue():
    global STARTED_LOOP

    if not STARTED_LOOP:
        print("Starting loop...")
        asyncio.create_task(looper())
        STARTED_LOOP = True

    args = request.args

    job_data = args["params"]

    data = json.loads(job_data)

    print(data)

    for job in data:

        if "id" not in job:
            print("No id")
            return "Missing id", 400

        if "parameters" not in job:
            print("No parameters")
            return "Missing parameters", 400

        if "worker" not in job:
            print("No worker")
            return "Missing worker", 400

        job["status"] = "pending"

        GLOBAL_QUEUE.append(job)

    return "OK", 200


@app.route("/api/v1/jobs")
async def list():

    light_results = [
        {
            "id": job["id"],
            "worker": job["worker"],
            "status": job["status"],
        }
        for job in GLOBAL_QUEUE
    ]

    for job in GLOBAL_QUEUE:
        if job["status"] == "complete":
            GLOBAL_QUEUE.remove(job)

    return jsonify(light_results)


@app.route("/api/v1/jobs/<job_id>")
async def get(job_id):

    for job in GLOBAL_QUEUE:
        if job["id"] == job_id:
            if job["status"] == "complete":
                GLOBAL_QUEUE.remove(job)

            return jsonify(job)

    return "Not found", 404


@app.route("/api/v1/jobs/<job_id>/image")
async def get_result(job_id):

    filename = "./results/{}.png".format(job_id)

    if not os.path.exists(filename):
        return "Not found", 404

    resp = await send_file(filename)

    # delete the file
    # os.remove(filename)

    return resp


async def looper():

    print("Starting loop...")

    while True:
        await asyncio.sleep(1)

        # Schedule new jobs
        pending_jobs = [job for job in GLOBAL_QUEUE if job["status"] == "pending"]
        num_pending_jobs = len(pending_jobs)

        print(f"{num_pending_jobs} pending jobs")

        while num_pending_jobs > 0:

            # get a job with that uid
            job = pending_jobs.pop()
            job["status"] = "scheduled"

            # send it to the worker
            worker_queue = OUT_QUEUES[job["worker"]]
            worker_queue.put(job)
            print(f"Scheduled {job['id']}")

            num_pending_jobs -= 1

        # Update progress
        for in_queue in IN_QUEUES:
            while not in_queue.empty():
                message = in_queue.get()
                job_id = message["id"]
                progress = message["progress"]
                status = message["status"]

                for job in GLOBAL_QUEUE:
                    if job["id"] == job_id:
                        job["progress"] = progress
                        job["status"] = status
                        break


if __name__ == "__main__":

    num_devices = torch.cuda.device_count()

    OUT_QUEUES = [multiprocessing.Queue() for _ in range(num_devices)]
    IN_QUEUES = [multiprocessing.Queue() for _ in range(num_devices)]

    # create results directory if it doesn't exist
    if not os.path.exists("./results"):
        os.makedirs("./results")

    for i in range(num_devices):
        p = multiprocessing.Process(target=worker, args=(OUT_QUEUES[i], IN_QUEUES[i]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p.start()

    app.run(host="0.0.0.0", port=42000, debug=True)

    while True:
        pass
