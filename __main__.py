import multiprocessing
from quart import Quart

app = Quart(__name__)

MESSAGE_QUEUE = None

@app.route("/v1/api/enqueue")
async def enqueue():
    pass

@app.route("/v1/api/jobs")
async def list():
    pass

@app.route("/v1/api/jobs/<job_id>")
async def get(job_id):
    pass

async def handler():
    pass

def worker(message_queue):
    while True:
        pass

if __name__ == "__main__":
    MESSAGE_QUEUE = multiprocessing.Queue()
    worker_process = multiprocessing.Process(target=worker, args=(MESSAGE_QUEUE,))
    worker_process.start()
    app.run(host="0.0.0.0", port=42000)