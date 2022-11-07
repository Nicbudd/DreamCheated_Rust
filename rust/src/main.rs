#![feature(portable_simd)]
use std::simd::*;
//use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use chrono::{DateTime, Utc};

fn update(maxRods: u16, maxPearls: u16, maxAttempts: u64, whenFound: &str, attempts: u64,
    totalExecTime: f64, speed: f64) {


    let writestring = format!("{},{},{},{},{},{},{}", maxRods, maxPearls, maxAttempts, whenFound, attempts, totalExecTime, speed);

    let mut file = File::create("dream.txt")
        .expect("Could not open the dream.txt file to write.");

    // Open a file in write-only mode, returns `io::Result<File>`
    file.write(writestring.as_bytes())
        .expect("Could not write to the dream.txt file.");

    println!("Attempts:{}, Rods: {}, Pearls: {}, Speed: {} attempts/sec", attempts, maxRods, maxPearls, speed);

}

fn main() {

    const PEARLTHRESHOLD: f64 = 20.0;
    const PEARLRANGE: f64 = 423.0;
    const GOLDTRADES: u16 = 262;
    const DREAMPEARLS: u16 = 42;

    const BLAZETHRESHOLD: f64 = 1.0;
    const BLAZERANGE: f64 = 2.0;
    const BLAZEKILLS: u16 = 305;
    const DREAMRODS: u16 = 211;

    const LOGFREQ: u64 = 100_000_000; // After how many attempts does it update the file


    let mut maxRods: u16 = 0;
    let mut maxPearls: u16 = 0;
    let mut maxAttempts: u64 = 0;

    let mut attempts: u64 = 0;
    let mut whenFound = String::new();
    let mut totalExecTime: f64 = 0.0;
    let mut speed: f64 = 0.0;


    // load previous data from file
    let contents = fs::read_to_string("dream.txt")
        .expect("Something went wrong reading the file");

    let split_string: Vec<&str> = contents.split(",").collect();

    maxRods = split_string[0].parse::<u16>().unwrap();
    maxPearls = split_string[1].parse::<u16>().unwrap();
    maxAttempts = split_string[2].parse::<u64>().unwrap();
    whenFound = String::from(split_string[3]);
    attempts = split_string[4].parse::<u64>().unwrap();
    totalExecTime = split_string[5].parse::<f64>().unwrap();
    speed = split_string[6].parse::<f64>().unwrap();


    // main loop
    loop {
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        //let mut frng = fastrand::seed(rng.gen());

        // start timer
        let now = SystemTime::now();

        for i in 0..LOGFREQ {

            // RODS
            let mut rodCount: u16 = 0;

            // super duper efficient way
            let list: [u64; 5] = [rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen::<u64>() & 0x1FFFF_FFFF_FFFFu64]; // 49 1s as a mask


            //let mut counted: u16 = 0;
            // why generate 305 random numbers when you can generate just 5?
            for item in list {
                rodCount += item.count_ones() as u16; // fast bit count go brrrrrrr
                /*if (rodCount + (305u16 - counted) < maxRods) { // if we can tell right away we aren't gonna get a max
                    continue;
                }*/
                //counted += 64;
            }


            // only do pearl section if rod section gets record (optimization)
            if rodCount >= maxRods {
                // PEARLS
                let mut pearlCount: u16 = 0;

                for j in 0..GOLDTRADES {
                    pearlCount += rng.gen_bool(PEARLTHRESHOLD / PEARLRANGE) as u16;
                }

                let newMax = (rodCount >= maxRods && pearlCount >= maxPearls) ||
                (rodCount >= DREAMRODS && pearlCount >= maxPearls) ||
                (rodCount >= maxRods && pearlCount >= DREAMPEARLS);

                if newMax {
                    let utc: DateTime<Utc> = Utc::now();
                    whenFound = utc.format("%F %T").to_string();
                    maxAttempts = attempts;
                    maxRods = rodCount;
                    maxPearls = pearlCount;
                    update(maxRods, maxPearls, maxAttempts, &whenFound, attempts, totalExecTime, speed);
                }
            }
            attempts += 1;
        }

        // stop timer
        match now.elapsed() {
            Ok(elapsed) => {
                let seconds = elapsed.as_secs_f64();
                speed = LOGFREQ as f64 / seconds;
                totalExecTime += seconds;
                update(maxRods, maxPearls, maxAttempts, &whenFound, attempts, totalExecTime, speed);
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}
