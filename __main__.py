import multiprocessing
from quart import Quart, request, jsonify, send_file
from worker import worker
import torch
import os
import asyncio
import random

app = Quart(__name__)

GLOBAL_QUEUE = []

OUT_QUEUES = None
IN_QUEUES = None


@app.route("/v1/api/enqueue")
async def enqueue():
    data = await request.get_json()

    for job in data:

        if "id" not in job:
            return "Missing id", 400

        if "parameters" not in job:
            return "Missing parameters", 400

        if "uid" not in job:
            return "Missing uid", 400

        data["status"] = "pending"

        GLOBAL_QUEUE.append(data)

    return "OK", 200


@app.route("/v1/api/jobs")
async def list():

    light_results = [
        {
            "id": job["id"],
            "status": job["status"],
        }
        for job in GLOBAL_QUEUE
    ]

    return jsonify(light_results)


@app.route("/v1/api/jobs/<job_id>")
async def get(job_id):

    for job in GLOBAL_QUEUE:
        if job["id"] == job_id:
            if job["status"] == "complete":
                GLOBAL_QUEUE.remove(job)

            return jsonify(job)

    return "Not found", 404


@app.route("/v1/api/jobs/<job_id>/image")
async def get_result(job_id):

    filename = "./results/{}.png".format(job_id)

    if not os.path.exists(filename):
        return "Not found", 404

    return send_file(filename)


async def looper():

    print("Starting loop...")

    while True:
        await asyncio.sleep(0.01)

        # Schedule new jobs
        pending_jobs = [job for job in GLOBAL_QUEUE if job["status"] == "pending"]
        num_pending_jobs = len(pending_jobs)

        while num_pending_jobs > 0:
            # get unique uids
            uids = [job["uid"] for job in pending_jobs]

            # get a random uid
            uid = random.choice(uids)

            # get a job with that uid
            job = [job for job in pending_jobs if job["uid"] == uid][0]
            job["status"] = "scheduled"

            # send it to a worker
            worker_queue = random.choice(OUT_QUEUES)
            worker_queue.put(job)

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

    for i in range(num_devices):
        p = multiprocessing.Process(target=worker, args=(OUT_QUEUES[i], IN_QUEUES[i]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p.start()

    # get default loop
    loop = asyncio.get_event_loop()
    # run the loop
    loop.run_until_complete(looper())

    app.run(host="0.0.0.0", port=42000)

    while True:
        pass
