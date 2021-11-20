import random, time
import datetime as dt

filename = "dream.txt"

batchSize = 100000

file = open(filename, "r")

#rods,pearls,attempts,execTime,dateTimeFound,speed
#0,0,0,0,0,0,0

data = ""
for line in file:
    data = line.split(",")

dreamMaxRods, dreamMaxPearls, maxAttempts, dateTimeFound, attempts, execTime, speed = data

dreamMaxRods = int(dreamMaxRods)
dreamMaxPearls = int(dreamMaxPearls)
maxAttempts = int(maxAttempts)
attempts = int(attempts)
execTime = float(execTime)

file.close()

startTime = time.time()

def update(startTime, execTime, batchTime):
    executionTime = (time.time() - startTime)
    execTime += executionTime
    startTime = time.time()
    speed = batchSize/batchTime

    data = [str(dreamMaxRods), str(dreamMaxPearls), str(maxAttempts), str(dateTimeFound), str(attempts), str(execTime), str(speed)]
    file = open(filename, "w")
    file.write(",".join(data))
    file.close()

    print(data)


batchExecutionStart = time.time()

while True:

    rods = 0
    pearls = 0

    for x in range(305):
        if random.random() < (1/2):
            rods += 1

    if rods >= dreamMaxRods:

        for x in range(262):
            if random.random() < (20/423):
                pearls += 1

        if (rods >= dreamMaxRods and pearls >= dreamMaxPearls) or (rods >= 211 and pearls >= dreamMaxPearls) or (rods >= dreamMaxRods and pearls >= 42):
            dreamMaxRods = rods
            dreamMaxPearls = pearls

            dateTimeFound = str(dt.datetime.now())[:-7]
            maxAttempts = attempts

            update(startTime, execTime, time.time() - startTime)

    attempts += 1

    if attempts % batchSize == 0:

        batchTime = time.time() - batchExecutionStart
        batchExecutionStart = time.time()

        update(startTime, execTime, batchTime)
