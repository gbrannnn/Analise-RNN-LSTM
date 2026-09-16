from worker import Worker
from pni import PNIDataCollector


if __name__ == "__main__":
    worker = Worker(collector=PNIDataCollector(limit=1000))  # You should replace None with an actual collector instance
    worker.run()
    


