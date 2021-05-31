import random, time
import datetime as dt



file = open("dream.txt", "r")

#rods,pearls,attempts,execTime,dateTimeFound
#0,0,0,0,0,0

data = ""
for line in file:
    data = line.split(",")

dreamMaxRods, dreamMaxPearls, maxAttempts, dateTimeFound, attempts, execTime = data

dreamMaxRods = int(dreamMaxRods)
dreamMaxPearls = int(dreamMaxPearls)
maxAttempts = int(maxAttempts)
attempts = int(attempts)
execTime = float(execTime)

file.close()

startTime = time.time()

def update(startTime, execTime):
    executionTime = (time.time() - startTime)
    execTime += executionTime
    startTime = time.time()

    data = [str(dreamMaxRods), str(dreamMaxPearls), str(maxAttempts), str(dateTimeFound), str(attempts), str(execTime)]
    file = open("dream.txt", "w")
    file.write(",".join(data))
    file.close()

    print(data)

while True:

    rods = 0
    pearls = 0

    for x in range(305):
        if random.random() < 0.5:
            rods += 1

    if rods >= dreamMaxRods:

        for x in range(262):
            if random.random() < (20/423):
                pearls += 1

        if (rods >= dreamMaxRods and pearls >= dreamMaxPearls) or (rods >= 211 and pearls >= dreamMaxPearls) or (rods >= dreamMaxRods and pearls >= 42):
            dreamMaxRods = rods
            dreamMaxPearls = pearls

            dateTimeFound = dt.datetime.now()
            maxAttempts = attempts

            update(startTime, execTime)

    attempts += 1

    if attempts % 100000 == 0:

        update(startTime, execTime)

