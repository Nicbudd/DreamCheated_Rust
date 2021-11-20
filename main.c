#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <sys/random.h>

void update(int maxRods, int maxPearls, unsigned long long maxAttempts, char whenFound[], unsigned long long attempts, double totalExecTime, double speed){
    // open file
    FILE *fp;
    if ((fp = fopen("dream.txt", "w")) == NULL){
        printf("File not found");
        exit(1);
    }

    // add data
    fprintf(fp, "%i,%i,%llu,%s,%llu,%lf,%lf", maxRods, maxPearls, maxAttempts, whenFound, attempts, totalExecTime, speed);

    // close file
    fclose(fp);

    printf("Attempts: %llu, Rods: %d, Pearls: %d, Speed: %lf attempts/sec\n", attempts, maxRods, maxPearls, speed);
}


int main()
{
    int maxRods;
    int maxPearls;
    unsigned long long maxAttempts;
    unsigned long long attempts;
    char whenFound[19];
    double totalExecTime;
    double speed;

    // open file
    FILE *fp;
    if ((fp = fopen("dream.txt", "r")) == NULL){
        printf("File not found");
        exit(1);
    }

    // find relevant data
    fscanf(fp, "%i,%i,%llu,%19[^\n],%llu,%lf,%lf", &maxRods, &maxPearls, &maxAttempts, whenFound, &attempts, &totalExecTime, &speed);

    // close file
    fclose(fp);


    printf("%i\n", maxRods);
    printf("%i\n", maxPearls);
    printf("%llu\n", maxAttempts);
    printf("%s\n", whenFound);
    printf("%llu\n", attempts);
    printf("%lf\n", totalExecTime);



    long seed;

    const int PEARLTHRESHOLD = 20;
    const int PEARLRANGE = 423;
    const int GOLDTRADES = 262;
    const int DREAMPEARLS = 42;


    const int BLAZETHRESHOLD = 1;
    const int BLAZERANGE = 2;
    const int BLAZEKILLS = 305;
    const int DREAMRODS = 211;


    const int LOGFREQ = 1000000;



    while (true) {

      seed = (long int)time(NULL);

      long int randSeed;
      getrandom(&seed, sizeof(seed), 0);

      seed ^= randSeed;

      printf("%ld\n", seed);
      srand((unsigned int)seed);

        // time length of batch of executions
        clock_t startT = clock();

        for (int i = 0; i < LOGFREQ; i++){

            //rod section
            int rodCount = 0;

            // for each of the blazes killed
            for (int j = 0; j < BLAZEKILLS; j++){
                int choice;

                // try to optimize 50/50 chance
                if (BLAZERANGE == 2){
                    choice = (rand() % 2) + 1; // random value between 1 and 2
                // if we're doing biased blaze drops:
                } else {
                    int r = rand();
                    choice = r / (((RAND_MAX / BLAZERANGE + 1)) + 1);
                }

                // if probability succeeds
                if (choice <= BLAZETHRESHOLD) {
                   rodCount++;
                }
            }

            // only do pearl section if rod section gets record (optimization)
            if (rodCount >= maxRods) {

                //pearl section
                int pearlCount = 0;

                for (int j = 0; j < GOLDTRADES; j++){
                    int r = (rand() / (RAND_MAX / PEARLRANGE + 1)) + 1 ;
                    if (r <= PEARLTHRESHOLD) {
                        pearlCount++;
                    }
                }


                // test for overall record
                // did we get a new maximum in anything?
                bool newMax = (rodCount >= maxRods && pearlCount >= maxPearls) ||
                (rodCount >= DREAMRODS && pearlCount >= maxPearls) ||
                (rodCount >= maxRods && pearlCount >= DREAMPEARLS);

                // if so, then update
                if (newMax) {
                    time_t t = time(NULL);
                    strftime(whenFound, sizeof(whenFound), "%F %T",
                      localtime(&t));
                    maxRods = rodCount;
                    maxPearls = pearlCount;
                    update(maxRods, maxPearls, maxAttempts, whenFound, attempts,
                      totalExecTime, 0);

                }
            }

            attempts++;

        }

        double seconds = ((double)(clock() - startT) / (double)CLOCKS_PER_SEC);

        speed = LOGFREQ / seconds;
        totalExecTime += seconds;

        update(maxRods, maxPearls, maxAttempts, whenFound, attempts, totalExecTime, speed);


        //printf("%lf attempts per second \n", LOGFREQ / seconds);

    }


    return 0;
}
