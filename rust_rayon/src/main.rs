//#![feature(portable_simd)]
//use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use chrono::{DateTime, Utc};
use rayon::prelude::*;

fn update(maxRods: u32, maxPearls: u32, maxAttempts: u64, whenFound: &str, attempts: u64,
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
    const GOLDTRADES: u32 = 262;
    const DREAMPEARLS: u32 = 42;

    const BLAZETHRESHOLD: f64 = 1.0;
    const BLAZERANGE: f64 = 2.0;
    const BLAZEKILLS: u32 = 305;
    const DREAMRODS: u32 = 211;

    const LOGFREQ: u64 = 100_000_000; // After how many attempts does it update the file


    let mut maxRods: u32 = 0;
    let mut maxPearls: u32 = 0;
    let mut maxAttempts: u64 = 0;

    let mut attempts: u64 = 0;
    let mut whenFound = String::new();
    let mut totalExecTime: f64 = 0.0;
    let mut speed: f64 = 0.0;

    let mut batch_size: u64 = 100_000;


    // load previous data from file
    let contents = fs::read_to_string("dream.txt")
        .expect("Something went wrong reading the file");

    let split_string: Vec<&str> = contents.split(",").collect();

    maxRods = split_string[0].parse::<u32>().unwrap();
    maxPearls = split_string[1].parse::<u32>().unwrap();
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

        if attempts >= 500_000_000 {
            batch_size = 500_000_000;
        } else if attempts >= 100_000_000 {
            batch_size = 100_000_000;
        } else if attempts >= 1_000_000 {
            batch_size = 1_000_000;
        }

        
        let rods: Vec<u32> = (0..batch_size).into_par_iter()
                                .map_init(
                                    || Xoshiro256PlusPlus::from_entropy(),
                                    |rng, _x| (rng.gen::<u128>(), rng.gen::<u128>(), rng.gen::<u64>() & 0x1FFFF_FFFF_FFFFu64),
                                )
                                .map(|x| x.0.count_ones() + x.1.count_ones() + x.2.count_ones())
                                .filter(|x| *x > maxRods || *x > DREAMRODS)
                                .collect();

        let mut pearls: Vec<(u32, u32)> = rods.clone().into_par_iter()
                                        .map_init(
                                            || Xoshiro256PlusPlus::from_entropy(),
                                            |rng, x| (x, (0..GOLDTRADES).into_iter()
                                                            .map(|_y| rng.gen_bool(PEARLTHRESHOLD / PEARLRANGE) as u32)
                                                            .sum()),
                                        )
                                        .filter(|x| (x.1 >= maxPearls || x.1 >= DREAMPEARLS))
                                        .collect();


        let newLen = pearls.len();
        
        if newLen > 0 {
            let mut newRecord;

            if newLen == 1 {
                newRecord = pearls.get(0).expect("Expected 1 item in list of length 1.");
            } else {
                println!("{} in batch higher than record.", newLen);
                pearls.sort_by(|b, a| a.0.partial_cmp(&b.0).unwrap());
                pearls.sort_by(|b, a| a.1.partial_cmp(&b.1).unwrap());

                newRecord = pearls.get(0).expect("Expected 1 item in list of length greater than 1.");
            }

            

            let utc: DateTime<Utc> = Utc::now();
            whenFound = utc.format("%F %T").to_string();
            maxAttempts = attempts;
            maxRods = newRecord.0;
            maxPearls = newRecord.1;

            println!("New record: {} rods, {} pearls", maxRods, maxPearls);
        }        
        
        attempts += batch_size;                               
                                        
        //dbg!(&rods);
        //dbg!(pearls);

        //std::process::exit(0);


        // stop timer
        match now.elapsed() {
            Ok(elapsed) => {
                let seconds = elapsed.as_secs_f64();
                speed = batch_size as f64 / seconds;
                totalExecTime += seconds;
                update(maxRods, maxPearls, maxAttempts, &whenFound, attempts, totalExecTime, speed);
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}
